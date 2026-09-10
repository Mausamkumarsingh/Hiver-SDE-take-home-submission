from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.agent.core import SupportAgent
from src.agent.schema import ActionType

def format_agent_response(resp, console: Console):
    """Renders structured agent output using Rich formatting."""
    action_color = "bold green" if resp.action == ActionType.AUTO_HANDLE else "bold red"
    
    console.print()
    console.print(Panel(
        f"[{action_color}]{resp.action.value}[/{action_color}] | "
        f"[bold cyan]Intent:[/] {resp.intent} (Confidence: [bold yellow]{resp.confidence:.2f}[/])\n"
        f"[bold white]Policy Reason:[/] {resp.reason}",
        title="[bold]SupportAgent Decision[/bold]",
        border_style="cyan"
    ))
    
    console.print(Panel(
        f"[italic white]{resp.reply}[/italic white]",
        title="[bold]Generated Customer Reply[/bold]",
        border_style="green" if resp.action == ActionType.AUTO_HANDLE else "yellow"
    ))
    
    if resp.evidence:
        table = Table(title="Retrieved Historical Resolutions (FAISS Evidence)", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Similarity", justify="center", width=12)
        table.add_column("Historical Customer Query", width=38)
        table.add_column("Historical Resolution", width=42)
        table.add_column("Links", width=25)
        
        for idx, ev in enumerate(resp.evidence, 1):
            links_str = "\n".join(ev.links) if ev.links else "[dim]None[/dim]"
            table.add_row(
                str(idx),
                f"{ev.similarity:.3f}",
                ev.query[:80] + "..." if len(ev.query) > 80 else ev.query,
                ev.resolution[:90] + "..." if len(ev.resolution) > 90 else ev.resolution,
                links_str
            )
        console.print(table)
    else:
        console.print("[yellow]No historical resolutions passed similarity threshold.[/yellow]")
    console.print()

def main():
    parser = argparse.ArgumentParser(description="Hiver AI Customer Support Agent Interactive CLI Demo")
    parser.add_argument("--query", "-q", type=str, help="Single query to process")
    parser.add_argument("--top-k", type=int, default=3, help="Number of historical resolutions to retrieve")
    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]====================================================[/bold blue]")
    console.print("[bold cyan]   Hiver AI Customer Support Agent (AmazonHelp)   [/bold cyan]")
    console.print("[bold blue]====================================================[/bold blue]")
    console.print("Initializing agent and loading FAISS index...")
    
    agent = SupportAgent()
    console.print("[bold green]Agent initialized successfully![/bold green]\n")

    if args.query:
        console.print(f"[bold]Input Query:[/] [italic]{args.query}[/italic]")
        resp = agent.process(args.query)
        format_agent_response(resp, console)
        return

    # Interactive Loop
    console.print("[bold yellow]Entering Interactive Demo Mode (Type 'exit' or 'quit' to stop)[/bold yellow]")
    while True:
        try:
            user_input = console.input("[bold green]Customer Message > [/bold green]")
            if user_input.strip().lower() in ["exit", "quit", "q"]:
                console.print("[bold cyan]Exiting CLI demo. Goodbye![/bold cyan]")
                break
            if not user_input.strip():
                continue
                
            resp = agent.process(user_input.strip())
            format_agent_response(resp, console)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]Session terminated.[/bold cyan]")
            break

if __name__ == "__main__":
    main()
