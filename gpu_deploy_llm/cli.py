"""Click CLI for GPU Deploy LLM.

Commands:
- web: Launch web dashboard
- health: Check shopper health
- inventory: List available GPU offers
- list-models: List supported models
- deploy: Deploy a model
- sessions: List active sessions
- status: Check session status
- extend: Extend a session
- costs: Check costs
- cleanup: Graceful session cleanup
- destroy: Force destroy session
- test-stress: Stress test
- test-orphan: Orphan simulation
- test-expiry: Expiry test
"""

import asyncio
import logging
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from gpu_deploy_llm import __version__
from gpu_deploy_llm.client import ShopperClient, SessionStatus
from gpu_deploy_llm.models import (
    list_models,
    calculate_requirements,
    Quantization,
    get_model_spec,
)
from gpu_deploy_llm.models.requirements import select_best_offer
from gpu_deploy_llm.ssh import SSHConnection
from gpu_deploy_llm.deploy import VLLMDeployer, DeploymentConfig, HealthChecker
from gpu_deploy_llm.diagnostics import DiagnosticCollector, Checkpoint
from gpu_deploy_llm.diagnostics.logger import setup_logging
from gpu_deploy_llm.utils.errors import (
    StaleInventoryError,
    NoAvailableOffersError,
    ProvisioningFailed,
)

console = Console()
logger = logging.getLogger(__name__)

# Default shopper URL
DEFAULT_SHOPPER_URL = "http://localhost:8080"


def run_async(coro):
    """Run async function in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--shopper-url",
    envvar="SHOPPER_URL",
    default=DEFAULT_SHOPPER_URL,
    help="Cloud-gpu-shopper URL",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.option(
    "--debug-shopper",
    is_flag=True,
    help="Enable verbose shopper API logging",
)
@click.pass_context
def cli(ctx, shopper_url: str, debug: bool, debug_shopper: bool):
    """GPU Deploy LLM - Deploy self-hosted LLMs on cloud GPUs."""
    ctx.ensure_object(dict)
    ctx.obj["shopper_url"] = shopper_url
    ctx.obj["debug"] = debug
    ctx.obj["debug_shopper"] = debug_shopper

    level = "DEBUG" if debug else "INFO"
    setup_logging(level=level, debug_shopper=debug_shopper)


@cli.command()
@click.option("--port", default=8080, help="Port to run dashboard on")
@click.pass_context
def web(ctx, port: int):
    """Launch web dashboard."""
    from gpu_deploy_llm.web.server import run_server

    console.print(f"[bold green]Starting web dashboard on http://localhost:{port}[/]")
    run_server(
        port=port,
        shopper_url=ctx.obj["shopper_url"],
        debug=ctx.obj["debug"],
    )


@cli.command()
@click.pass_context
def health(ctx):
    """Check shopper health and readiness."""

    async def _health():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            # Check health
            console.print("Checking shopper health...")
            try:
                health_resp = await client.health_check()
                console.print(f"  [green]/health: OK[/] - {health_resp.status}")
            except Exception as e:
                console.print(f"  [red]/health: FAILED[/] - {e}")
                return

            # Check ready
            try:
                ready_resp = await client.ready_check()
                if ready_resp.ready:
                    console.print(f"  [green]/ready: OK[/]")
                else:
                    console.print(f"  [yellow]/ready: Not ready[/] - {ready_resp.message}")
            except Exception as e:
                console.print(f"  [red]/ready: FAILED[/] - {e}")

    run_async(_health())


@cli.command()
@click.option("--min-vram", type=float, help="Minimum VRAM in GB")
@click.option("--max-price", type=float, help="Maximum hourly price")
@click.option("--provider", help="Filter by provider (vastai, tensordock)")
@click.option("--gpu-type", help="Filter by GPU type")
@click.option("--min-gpu-count", type=int, help="Minimum GPU count")
@click.pass_context
def inventory(
    ctx,
    min_vram: Optional[float],
    max_price: Optional[float],
    provider: Optional[str],
    gpu_type: Optional[str],
    min_gpu_count: Optional[int],
):
    """List available GPU offers."""

    async def _inventory():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            # Wait for shopper to be ready
            console.print("Waiting for shopper to be ready...")
            await client.wait_for_ready()

            console.print("Querying inventory...")
            offers = await client.get_inventory(
                min_vram=min_vram,
                max_price=max_price,
                provider=provider,
                gpu_type=gpu_type,
                min_gpu_count=min_gpu_count,
            )

            if not offers:
                console.print("[yellow]No offers found matching criteria[/]")
                return

            # Display as table
            table = Table(title=f"Available GPU Offers ({len(offers)})")
            table.add_column("ID", style="dim")
            table.add_column("Provider")
            table.add_column("GPU")
            table.add_column("Count", justify="right")
            table.add_column("VRAM", justify="right")
            table.add_column("$/hr", justify="right")
            table.add_column("Region")

            for offer in offers:
                table.add_row(
                    offer.id[:12] + "...",
                    offer.provider,
                    offer.gpu_type,
                    str(offer.gpu_count),
                    f"{offer.vram_gb:.0f}GB",
                    f"${offer.price_per_hour:.3f}",
                    offer.region or "-",
                )

            console.print(table)

    run_async(_inventory())


@cli.command("list-models")
def list_models_cmd():
    """List supported LLM models."""
    models = list_models()

    table = Table(title="Supported Models")
    table.add_column("Model ID")
    table.add_column("Name")
    table.add_column("FP16 VRAM")
    table.add_column("4-bit VRAM")
    table.add_column("Description")

    for model in models:
        table.add_row(
            model.model_id,
            model.name,
            f"{model.vram_fp16_gb}GB",
            f"{model.vram_4bit_gb}GB",
            model.description,
        )

    console.print(table)


@cli.command()
@click.argument("model_id")
@click.option(
    "--quantization",
    type=click.Choice(["none", "awq", "gptq"]),
    default="none",
    help="Quantization method",
)
@click.option("--max-price", type=float, default=1.0, help="Maximum hourly price")
@click.option("--reservation-hours", type=int, default=1, help="Reservation duration (1-12)")
@click.option("--provider", help="Preferred provider (vastai, tensordock)")
@click.option("--skip-verification", is_flag=True, help="Skip deployment verification")
@click.pass_context
def deploy(
    ctx,
    model_id: str,
    quantization: str,
    max_price: float,
    reservation_hours: int,
    provider: Optional[str],
    skip_verification: bool,
):
    """Deploy a model on cloud GPU."""
    quant = Quantization(quantization)

    async def _deploy():
        # Calculate requirements
        console.print(f"[bold]Deploying {model_id}[/]")

        try:
            requirements = calculate_requirements(model_id, quant)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/]")
            return

        console.print(
            f"  Requirements: {requirements.min_vram_gb}GB VRAM, "
            f"{requirements.min_gpu_count} GPU(s)"
        )

        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            # Wait for shopper
            with console.status("Waiting for shopper..."):
                await client.wait_for_ready()

            # Query inventory with retry on stale offers
            excluded_offers = set()
            session_response = None
            ssh_key = None

            for attempt in range(3):
                # Get offers
                with console.status("Querying inventory..."):
                    offers = await client.get_inventory(
                        min_vram=requirements.total_vram_needed,
                        max_price=max_price,
                        provider=provider,
                        min_gpu_count=requirements.min_gpu_count,
                    )

                # Filter excluded offers
                offers = [o for o in offers if o.id not in excluded_offers]

                if not offers:
                    console.print("[red]No suitable GPU offers found[/]")
                    return

                # Select best offer
                offer = select_best_offer(offers, requirements, max_price)
                if not offer:
                    console.print("[red]No offers meet requirements[/]")
                    return

                console.print(
                    f"  Selected: {offer.gpu_type} @ ${offer.price_per_hour:.3f}/hr "
                    f"({offer.provider})"
                )

                # Create session
                try:
                    with console.status("Creating session..."):
                        session_response = await client.create_session(
                            offer_id=offer.id,
                            reservation_hours=reservation_hours,
                        )
                        # CRITICAL: Capture SSH key immediately!
                        ssh_key = session_response.ssh_private_key
                        console.print(
                            f"  Session created: {session_response.session.id}"
                        )
                        break

                except StaleInventoryError as e:
                    console.print(
                        f"[yellow]Offer stale, trying another... (attempt {attempt + 1}/3)[/]"
                    )
                    excluded_offers.add(e.offer_id)
                    continue

            if not session_response or not ssh_key:
                console.print("[red]Failed to create session after 3 attempts[/]")
                return

            session = session_response.session

            # Initialize diagnostics
            collector = DiagnosticCollector(
                session_id=session.id,
                model_id=requirements.model_id,
                provider=session.provider,
                gpu_type=session.gpu_type,
                gpu_count=session.gpu_count,
            )

            await collector.collect(Checkpoint.SESSION_CREATED, shopper_client=client)

            try:
                # Wait for session to be running
                with console.status("Waiting for session to be running..."):
                    session = await client.wait_for_running(
                        session.id,
                        provider=session.provider,
                    )

                console.print(
                    f"  SSH: {session.ssh_user}@{session.ssh_host}:{session.ssh_port}"
                )

                await collector.collect(Checkpoint.SESSION_RUNNING, shopper_client=client)

                # Connect SSH
                console.print("Connecting via SSH...")
                async with SSHConnection(
                    host=session.ssh_host,
                    port=session.ssh_port,
                    user=session.ssh_user,
                    private_key=ssh_key,
                ) as ssh:
                    console.print("  [green]SSH connected[/]")

                    await collector.collect(
                        Checkpoint.SSH_CONNECTED,
                        shopper_client=client,
                        ssh=ssh,
                    )

                    # Deploy vLLM
                    await collector.collect(
                        Checkpoint.DEPLOYMENT_STARTED,
                        shopper_client=client,
                        ssh=ssh,
                    )

                    deployer = VLLMDeployer(ssh)
                    config = DeploymentConfig(
                        model_id=requirements.model_id,
                        quantization=quant,
                        gpu_count=session.gpu_count,
                    )

                    with console.status("Deploying vLLM (this may take several minutes)..."):
                        deploy_result = await deployer.deploy(config)

                    if not deploy_result.success:
                        console.print(f"[red]Deployment failed: {deploy_result.error}[/]")
                        await collector.collect(
                            Checkpoint.ERROR,
                            shopper_client=client,
                            ssh=ssh,
                            error=deploy_result.error,
                        )
                        return

                    console.print(f"  [green]vLLM deployed[/]")

                    # Wait for model to load
                    if not skip_verification:
                        checker = HealthChecker(
                            ssh=ssh,
                            endpoint=deploy_result.endpoint,
                            api_key=deploy_result.api_key,
                        )

                        with console.status("Waiting for model to load..."):
                            try:
                                await checker.wait_for_ready(timeout=300)
                            except TimeoutError:
                                console.print("[yellow]Model loading timed out[/]")

                        # Run verification
                        console.print("Running verification checks...")
                        verification = await checker.verify_all()

                        for check in verification.checks:
                            status_color = "green" if check.status.value == "passed" else "red"
                            console.print(
                                f"  [{status_color}]{check.name}: {check.status.value}[/] - {check.message}"
                            )

                    await collector.collect(
                        Checkpoint.DEPLOYMENT_COMPLETE,
                        shopper_client=client,
                        ssh=ssh,
                    )

                # Display results
                console.print()
                panel = Panel(
                    f"""[bold green]Deployment Complete[/]

[bold]Endpoint:[/] {deploy_result.endpoint}
[bold]API Key:[/] {deploy_result.api_key}
[bold]Session:[/] {session.id}
[bold]Provider:[/] {session.provider}
[bold]GPU:[/] {session.gpu_type} x{session.gpu_count}
[bold]Cost:[/] ${session.price_per_hour:.3f}/hr

[bold]Test command:[/]
curl -H "Authorization: Bearer {deploy_result.api_key}" \\
  {deploy_result.endpoint}/models

[bold]Cleanup:[/]
gpu-deploy-llm cleanup {session.id}""",
                    title="Deployment Summary",
                )
                console.print(panel)

            except Exception as e:
                console.print(f"[red]Error: {e}[/]")
                await collector.collect(
                    Checkpoint.ERROR,
                    shopper_client=client,
                    error=str(e),
                )

                # Cleanup on error
                console.print("Cleaning up session...")
                try:
                    await client.force_destroy(session.id)
                except Exception:
                    pass

    run_async(_deploy())


@cli.command()
@click.option("--status", "filter_status", help="Filter by status")
@click.option("--consumer-id", help="Filter by consumer ID")
@click.pass_context
def sessions(ctx, filter_status: Optional[str], consumer_id: Optional[str]):
    """List sessions."""

    async def _sessions():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            sessions_list = await client.list_sessions(
                status=filter_status,
                consumer_id=consumer_id,
            )

            if not sessions_list:
                console.print("[yellow]No sessions found[/]")
                return

            table = Table(title=f"Sessions ({len(sessions_list)})")
            table.add_column("ID")
            table.add_column("Status")
            table.add_column("Provider")
            table.add_column("GPU")
            table.add_column("Created")
            table.add_column("$/hr")

            for s in sessions_list:
                status_color = {
                    SessionStatus.RUNNING: "green",
                    SessionStatus.PROVISIONING: "yellow",
                    SessionStatus.PENDING: "yellow",
                    SessionStatus.FAILED: "red",
                    SessionStatus.STOPPED: "dim",
                }.get(s.status, "white")

                table.add_row(
                    s.id[:16],
                    f"[{status_color}]{s.status.value}[/]",
                    s.provider,
                    s.gpu_type,
                    s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "-",
                    f"${s.price_per_hour:.3f}",
                )

            console.print(table)

    run_async(_sessions())


@cli.command()
@click.argument("session_id")
@click.pass_context
def status(ctx, session_id: str):
    """Check session status and diagnostics."""

    async def _status():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            # Get session
            session = await client.get_session(session_id)

            console.print(f"[bold]Session: {session.id}[/]")
            console.print(f"  Status: {session.status.value}")
            console.print(f"  Provider: {session.provider}")
            console.print(f"  GPU: {session.gpu_type} x{session.gpu_count}")
            console.print(f"  Price: ${session.price_per_hour:.3f}/hr")

            if session.ssh_host:
                console.print(f"  SSH: {session.ssh_user}@{session.ssh_host}:{session.ssh_port}")

            if session.expires_at:
                console.print(f"  Expires: {session.expires_at}")

            if session.error:
                console.print(f"  [red]Error: {session.error}[/]")

            # Get diagnostics
            try:
                diag = await client.get_session_diagnostics(session_id)
                console.print(f"\n[bold]Diagnostics:[/]")
                console.print(f"  SSH Available: {diag.ssh_available}")
                if diag.uptime:
                    console.print(f"  Uptime: {diag.uptime}")
                if diag.time_to_expiry:
                    console.print(f"  Time to Expiry: {diag.time_to_expiry}")
            except Exception:
                pass

    run_async(_status())


@cli.command()
@click.argument("session_id")
@click.option("--hours", type=int, default=1, help="Hours to extend (1-12)")
@click.pass_context
def extend(ctx, session_id: str, hours: int):
    """Extend a running session."""

    async def _extend():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            session = await client.extend_session(session_id, hours)
            console.print(f"[green]Session extended[/]")
            console.print(f"  New expiry: {session.expires_at}")

    run_async(_extend())


@cli.command()
@click.option("--session", "session_id", help="Filter by session ID")
@click.option("--daily", is_flag=True, help="Show daily costs")
@click.pass_context
def costs(ctx, session_id: Optional[str], daily: bool):
    """Check costs."""

    async def _costs():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            cost_info = await client.get_costs(session_id=session_id)

            console.print(f"[bold]Costs[/]")
            console.print(f"  Total: ${cost_info.total_cost:.4f}")
            console.print(f"  Duration: {cost_info.duration_hours:.2f} hours")

            if cost_info.breakdown:
                console.print(f"\n[bold]Breakdown:[/]")
                for item in cost_info.breakdown:
                    console.print(f"  {item}")

    run_async(_costs())


@cli.command()
@click.argument("session_id")
@click.pass_context
def cleanup(ctx, session_id: str):
    """Graceful session cleanup (signal done)."""

    async def _cleanup():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            session = await client.signal_done(session_id)
            console.print(f"[green]Cleanup initiated[/]")
            console.print(f"  Status: {session.status.value}")

    run_async(_cleanup())


@cli.command()
@click.argument("session_id")
@click.pass_context
def destroy(ctx, session_id: str):
    """Force destroy session (immediate cleanup)."""

    async def _destroy():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            session = await client.force_destroy(session_id)
            console.print(f"[green]Session destroyed[/]")
            console.print(f"  Status: {session.status.value}")

    run_async(_destroy())


@cli.command("test-stress")
@click.option("--cycles", type=int, default=5, help="Number of create/destroy cycles")
@click.option("--provider", help="Provider to test")
@click.pass_context
def test_stress(ctx, cycles: int, provider: Optional[str]):
    """Stress test: rapid create/destroy cycles."""
    console.print(f"[bold]Stress test: {cycles} cycles[/]")

    async def _stress():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            await client.wait_for_ready()

            for i in range(cycles):
                console.print(f"\n[bold]Cycle {i + 1}/{cycles}[/]")

                # Get cheapest offer
                offers = await client.get_inventory(
                    min_vram=4,
                    max_price=0.50,
                    provider=provider,
                )

                if not offers:
                    console.print("[yellow]No offers available[/]")
                    continue

                offer = min(offers, key=lambda o: o.price_per_hour)

                # Create session
                try:
                    response = await client.create_session(
                        offer_id=offer.id,
                        reservation_hours=1,
                    )
                    console.print(f"  Created: {response.session.id}")

                    # Immediately destroy
                    await client.force_destroy(response.session.id)
                    console.print(f"  Destroyed")

                except Exception as e:
                    console.print(f"  [red]Error: {e}[/]")

            console.print(f"\n[green]Stress test complete[/]")

    run_async(_stress())


@cli.command("test-orphan")
@click.option("--provider", help="Provider to test")
@click.pass_context
def test_orphan(ctx, provider: Optional[str]):
    """Orphan simulation: create session, don't cleanup."""
    console.print("[bold]Orphan simulation[/]")
    console.print("[yellow]Warning: This creates a session without cleanup[/]")

    async def _orphan():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            await client.wait_for_ready()

            offers = await client.get_inventory(
                min_vram=4,
                max_price=0.50,
                provider=provider,
            )

            if not offers:
                console.print("[red]No offers available[/]")
                return

            offer = min(offers, key=lambda o: o.price_per_hour)

            response = await client.create_session(
                offer_id=offer.id,
                reservation_hours=1,
            )

            console.print(f"[yellow]Created orphan session: {response.session.id}[/]")
            console.print("Session will NOT be cleaned up - testing shopper orphan detection")

    run_async(_orphan())


@cli.command("test-expiry")
@click.option("--hours", type=int, default=1, help="Reservation hours")
@click.pass_context
def test_expiry(ctx, hours: int):
    """Expiry test: let session expire naturally."""
    console.print(f"[bold]Expiry test: {hours} hour reservation[/]")

    async def _expiry():
        async with ShopperClient(
            ctx.obj["shopper_url"],
            debug=ctx.obj["debug_shopper"],
        ) as client:
            await client.wait_for_ready()

            offers = await client.get_inventory(min_vram=4, max_price=0.50)

            if not offers:
                console.print("[red]No offers available[/]")
                return

            offer = min(offers, key=lambda o: o.price_per_hour)

            response = await client.create_session(
                offer_id=offer.id,
                reservation_hours=hours,
            )

            console.print(f"[yellow]Created session: {response.session.id}[/]")
            console.print(f"Session will expire in {hours} hour(s)")
            console.print("Testing shopper automatic expiry/cleanup")

    run_async(_expiry())


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
