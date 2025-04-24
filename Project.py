import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
from tabulate import tabulate
from collections import Counter


class PremierLeagueNetwork:
    def __init__(self, dataset_path="Combined_PremierLeague.csv"):
        self.dataset_path = dataset_path
        self.players_df = None
        self.G = nx.Graph()
        self.player_col = "name"
        self.club_col = "club"
        self.season_col = "Season"
        self.all_players = set()
        self.season_networks = {}
        self.club_networks = {}

    def load_dataset(self):
        try:
            self.players_df = pd.read_csv(self.dataset_path)
            self.players_df[self.player_col] = (
                self.players_df[self.player_col].astype(str).str.strip()
            )
            self.players_df[self.club_col] = (
                self.players_df[self.club_col].astype(str).str.strip()
            )
            self.all_players = set(self.players_df[self.player_col].unique())
            print(f"Loaded {len(self.players_df)} records from '{self.dataset_path}'.")
            print(f"Found {len(self.all_players)} unique players across all seasons.")
            return True
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return False

    def build_teammate_network(self):
        if self.players_df is None:
            print("No dataset loaded.")
            return False

        df = self.players_df.dropna(
            subset=[self.player_col, self.club_col, self.season_col]
        )
        self.G.clear()
        self.season_networks = {}
        self.club_networks = {}

        all_clubs = set(df[self.club_col].unique())

        # Initialize season networks
        for season in df[self.season_col].unique():
            self.season_networks[season] = nx.Graph()
            season_df = df[df[self.season_col] == season]
            self.season_networks[season].add_nodes_from(
                season_df[self.player_col].unique()
            )

        # Add all players as nodes
        self.G.add_nodes_from(df[self.player_col].unique())

        # Create club networks
        club_graph = nx.Graph()
        club_graph.add_nodes_from(all_clubs)
        shared_players_graph = nx.Graph()
        shared_players_graph.add_nodes_from(all_clubs)

        # Build player connections by season and club
        for season in df[self.season_col].unique():
            season_df = df[df[self.season_col] == season]
            for club, club_df in season_df.groupby(self.club_col):
                players = sorted(set(club_df[self.player_col]))
                # Add connections between teammates
                for p1, p2 in itertools.combinations(players, 2):
                    if self.G.has_edge(p1, p2):
                        self.G[p1][p2]["weight"] += 1
                        self.G[p1][p2]["history"].append((club, season))
                    else:
                        self.G.add_edge(p1, p2, weight=1, history=[(club, season)])

                    # Add to season network
                    season_graph = self.season_networks[season]
                    if not season_graph.has_edge(p1, p2):
                        season_graph.add_edge(p1, p2, club=club)

        # Build club connection data based on player movements
        player_club_history = {}
        for _, row in df.iterrows():
            player = row[self.player_col]
            club = row[self.club_col]
            season = row[self.season_col]

            if player not in player_club_history:
                player_club_history[player] = []
            player_club_history[player].append((club, season))

        # Create club network edges
        club_connections = {}
        for player, history in player_club_history.items():
            if len(history) <= 1:
                continue

            # Sort by season
            history.sort(key=lambda x: x[1])

            # Create connections between consecutive clubs
            for i in range(len(history) - 1):
                club1, season1 = history[i]
                club2, season2 = history[i + 1]

                if club1 == club2:
                    continue

                edge = tuple(sorted([club1, club2]))
                if edge not in club_connections:
                    club_connections[edge] = {"weight": 0, "players": []}

                club_connections[edge]["weight"] += 1
                club_connections[edge]["players"].append(player)

        # Add connections to club graph
        for (club1, club2), data in club_connections.items():
            club_graph.add_edge(
                club1, club2, weight=data["weight"], players=data["players"]
            )

        self.club_networks["transfers"] = club_graph

        # Create shared players network
        for player, clubs_seasons in player_club_history.items():
            clubs = set([club for club, _ in clubs_seasons])
            if len(clubs) <= 1:
                continue

            for club1, club2 in itertools.combinations(clubs, 2):
                if shared_players_graph.has_edge(club1, club2):
                    shared_players_graph[club1][club2]["weight"] += 1
                    shared_players_graph[club1][club2]["players"].append(player)
                else:
                    shared_players_graph.add_edge(
                        club1, club2, weight=1, players=[player]
                    )

        self.club_networks["shared_players"] = shared_players_graph

        print(
            f"Network built with {self.G.number_of_nodes()} players and {self.G.number_of_edges()} connections"
        )
        return True

    def find_shortest_path(self, player1, player2):
        if player1 not in self.G or player2 not in self.G:
            print(f"One or both players not found in network.")
            return None

        try:
            path = nx.shortest_path(self.G, source=player1, target=player2)
            print(f"\n🔗 Connection between {player1} and {player2}:")
            print(f"Degree of separation: {len(path) - 1}")

            print("\nDetailed path with context:")
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                seasons = self.G[p1][p2].get("history", [])
                season_text = ", ".join(
                    [f"{club} ({season})" for club, season in seasons]
                )
                print(f"{i + 1}. {p1} ↔ {p2} via: {season_text}")

            return path
        except nx.NetworkXNoPath:
            print(f"No connection between {player1} and {player2}.")
            return None

    def find_most_connected_players(self, top_n=10, by_season=None):
        if self.G.number_of_nodes() == 0:
            print("Network not built yet.")
            return

        if by_season:
            if by_season not in self.season_networks:
                print(f"Season {by_season} not found.")
                return
            season_graph = self.season_networks[by_season]
            degrees = sorted(
                [(node, season_graph.degree(node)) for node in season_graph.nodes()],
                key=lambda x: x[1],
                reverse=True,
            )
            print(f"\nMost connected players in {by_season} season:")
        else:
            degrees = sorted(
                [(node, self.G.degree(node)) for node in self.G.nodes()],
                key=lambda x: x[1],
                reverse=True,
            )
            print("\nMost connected players across all seasons:")

        headers = ["Rank", "Player", "Unique Teammates"]
        table_data = [
            (i + 1, player, degree)
            for i, (player, degree) in enumerate(degrees[:top_n])
        ]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        return degrees[:top_n]

    def visualize_network(self, player=None, depth=1, save_path=None):
        if self.G.number_of_nodes() == 0:
            print("Network not built yet.")
            return

        plt.figure(figsize=(16, 12))

        if player and player in self.G:
            # Get ego network
            ego_nodes = {player}
            current_nodes = {player}

            for _ in range(depth):
                next_nodes = set()
                for node in current_nodes:
                    next_nodes.update(self.G.neighbors(node))
                ego_nodes.update(next_nodes)
                current_nodes = next_nodes

            subgraph = self.G.subgraph(ego_nodes)

            if len(subgraph.nodes()) > 50:
                print(
                    f"Filtering network to show only important connections (network had {len(subgraph.nodes())} nodes)"
                )
                central_edges = [
                    (player, n, d["weight"]) for n, d in subgraph[player].items()
                ]
                central_edges.sort(key=lambda x: x[2], reverse=True)
                keep_nodes = {player} | {edge[1] for edge in central_edges[:30]}

                for node in list(keep_nodes - {player}):
                    node_edges = [
                        (node, n, d["weight"])
                        for n, d in subgraph[node].items()
                        if n in keep_nodes
                    ]
                    node_edges.sort(key=lambda x: x[2], reverse=True)
                    keep_nodes |= {edge[1] for edge in node_edges[:5]}

                subgraph = subgraph.subgraph(keep_nodes)

            pos = nx.kamada_kawai_layout(subgraph)

            edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
            max_weight = max(edge_weights) if edge_weights else 1

            nx.draw_networkx_edges(
                subgraph,
                pos,
                width=[0.5 + w / max_weight * 3 for w in edge_weights],
                alpha=0.6,
                edge_color="gray",
            )

            node_sizes = [300 + 100 * subgraph.degree(n) for n in subgraph.nodes()]
            node_colors = [
                "red" if n == player else "lightblue" for n in subgraph.nodes()
            ]

            nx.draw_networkx_nodes(
                subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8
            )

            label_pos = {
                node: (pos[node][0], pos[node][1] - 0.02) for node in subgraph.nodes()
            }
            nx.draw_networkx_labels(
                subgraph,
                label_pos,
                font_size=9,
                font_weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

            plt.title(f"Network around {player} (Depth: {depth})", fontsize=16)
        else:
            # Full network visualization (limited to most connected)
            if self.G.number_of_nodes() > 50:
                print(
                    "Full graph is too large to visualize. Showing top connected players."
                )
                degrees = sorted(
                    [(node, self.G.degree(node)) for node in self.G.nodes()],
                    key=lambda x: x[1],
                    reverse=True,
                )
                top_players = [player for player, _ in degrees[:40]]
                subgraph = self.G.subgraph(top_players)
            else:
                subgraph = self.G

            pos = nx.kamada_kawai_layout(subgraph)

            node_sizes = [300 + 30 * subgraph.degree(n) for n in subgraph.nodes()]
            nx.draw_networkx_nodes(
                subgraph, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8
            )
            nx.draw_networkx_edges(
                subgraph, pos, width=0.7, alpha=0.4, edge_color="gray"
            )
            nx.draw_networkx_labels(
                subgraph,
                pos,
                font_size=9,
                font_weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
            )

            plt.title("Premier League Player Network", fontsize=16)

        plt.tight_layout()
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Network visualization saved to {save_path}")

        plt.show()

    def visualize_club_network(
        self, network_type="transfers", top_n=20, save_path=None
    ):
        if network_type not in self.club_networks:
            print(f"Club network type '{network_type}' not found.")
            return

        graph = self.club_networks[network_type]

        if graph.number_of_nodes() == 0:
            print("Club network not built yet.")
            return

        # Filter to top clubs by connection weight if needed
        if graph.number_of_nodes() > top_n:
            club_weights = {}
            for club in graph.nodes():
                total_weight = sum(
                    data["weight"] for _, _, data in graph.edges(club, data=True)
                )
                club_weights[club] = total_weight

            top_clubs = sorted(club_weights.items(), key=lambda x: x[1], reverse=True)[
                :top_n
            ]
            top_club_names = [club for club, _ in top_clubs]
            subgraph = graph.subgraph(top_club_names)
        else:
            subgraph = graph

        plt.figure(figsize=(18, 14))

        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)

        edge_weights = [subgraph[u][v]["weight"] for u, v in subgraph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        # Calculate node sizes based on connection strength
        node_weights = {}
        for club in subgraph.nodes():
            node_weights[club] = sum(
                data["weight"] for _, _, data in subgraph.edges(club, data=True)
            )
        max_node_weight = max(node_weights.values()) if node_weights else 1
        node_sizes = [
            1000 * (0.3 + 0.7 * node_weights[n] / max_node_weight)
            for n in subgraph.nodes()
        ]

        edge_colors = [
            subgraph[u][v]["weight"] / max_weight for u, v in subgraph.edges()
        ]

        edges = nx.draw_networkx_edges(
            subgraph,
            pos,
            width=[1 + 5 * w / max_weight for w in edge_weights],
            alpha=0.7,
            edge_color=edge_colors,
            edge_cmap=plt.cm.YlOrRd,
        )

        nodes = nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_size=node_sizes,
            node_color="skyblue",
            edgecolors="black",
            alpha=0.9,
        )

        nx.draw_networkx_labels(
            subgraph,
            pos,
            font_size=10,
            font_weight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
        )

        if network_type == "transfers":
            title_text = "Club Transfer Network"
            weight_meaning = "players transferred"
        else:
            title_text = "Club Shared Players Network"
            weight_meaning = "shared players"

        edge_labels = {
            (u, v): f"{d['weight']} {weight_meaning}"
            for u, v, d in subgraph.edges(data=True)
            if d["weight"] >= max_weight * 0.5
        }
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        plt.title(f"{title_text} (Top {len(subgraph.nodes())} Clubs)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Club network visualization saved to {save_path}")

        plt.show()

    def visualize_player_club_movements(self, player=None, top_n=10, save_path=None):
        if self.players_df is None:
            print("No dataset loaded.")
            return

        # Prepare data
        df = self.players_df.dropna(
            subset=[self.player_col, self.club_col, self.season_col]
        )

        # Group by player and get club history
        player_movements = {}
        player_df_grouped = df.sort_values(self.season_col).groupby(self.player_col)

        for player_name, group in player_df_grouped:
            club_history = list(zip(group[self.club_col], group[self.season_col]))
            transitions = []
            prev_club = None
            prev_season = None

            for club, season in club_history:
                if prev_club is not None and club != prev_club:
                    transitions.append((prev_club, club, prev_season, season))
                prev_club = club
                prev_season = season

            if transitions:
                player_movements[player_name] = transitions

        if not player_movements:
            print("No player movements found in the dataset.")
            return

        plt.figure(figsize=(16, 10))

        if player and player in player_movements:
            # Visualize specific player's movements
            self._plot_player_movements(player, player_movements[player])

        else:
            # Show players with most club changes
            players_by_moves = sorted(
                [(name, len(moves)) for name, moves in player_movements.items()],
                key=lambda x: x[1],
                reverse=True,
            )

            display_n = min(top_n, len(players_by_moves))
            fig, axes = plt.subplots(nrows=display_n, figsize=(16, 4 * display_n))

            if display_n == 1:
                axes = [axes]

            plt.subplots_adjust(hspace=0.4)

            for i, (player_name, num_moves) in enumerate(players_by_moves[:display_n]):
                plt.sca(axes[i])
                self._plot_player_movements(player_name, player_movements[player_name])

            plt.suptitle(f"Top {display_n} Players by Club Movements", fontsize=16)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Player movement visualization saved to {save_path}")

        plt.show()

    def _plot_player_movements(self, player_name, movements):
        """Helper method to plot a single player's movements between clubs"""
        # Extract all clubs
        all_clubs = set()
        for src, dst, src_season, dst_season in movements:
            all_clubs.add(src)
            all_clubs.add(dst)

        # Create positions for clubs
        clubs_list = sorted(list(all_clubs))
        club_positions = {club: i for i, club in enumerate(clubs_list)}

        # Plot clubs as nodes
        plt.scatter(
            list(club_positions.values()),
            [0] * len(club_positions),
            s=300,
            color="skyblue",
            edgecolor="black",
            zorder=2,
        )

        # Add club labels
        for club, pos in club_positions.items():
            plt.text(
                pos,
                0.1,
                club,
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )

        # Plot movements as arcs
        for src_club, dst_club, src_season, dst_season in movements:
            src_pos = club_positions[src_club]
            dst_pos = club_positions[dst_club]

            # Create arc between clubs
            arc_height = 0.5

            x = np.linspace(src_pos, dst_pos, 100)
            y = arc_height * np.sin(np.linspace(0, np.pi, 100))

            plt.plot(x, y, "r-", alpha=0.7, linewidth=2, zorder=1)

            # Add season label
            mid_x = (src_pos + dst_pos) / 2
            plt.text(
                mid_x,
                arc_height * 1.1,
                f"{src_season} → {dst_season}",
                ha="center",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        plt.title(f"Club Movements: {player_name}", fontsize=14)
        plt.axis("off")
        plt.ylim(-0.2, 1.2)
        plt.xlim(-0.5, max(club_positions.values()) + 0.5)

    def get_player_suggestions(self, partial_name, max_suggestions=5):
        if not partial_name:
            return []

        partial_name = partial_name.lower()
        matches = [
            player for player in self.all_players if partial_name in player.lower()
        ]
        return sorted(
            matches, key=lambda x: (not x.lower().startswith(partial_name), len(x))
        )[:max_suggestions]

    def get_player_stats(self, player_name):
        if self.players_df is None:
            print("No dataset loaded.")
            return

        player_data = self.players_df[self.players_df[self.player_col] == player_name]

        if player_data.empty:
            print(f"No data found for {player_name}")
            return

        seasons = player_data[self.season_col].unique()
        clubs = player_data[self.club_col].unique()

        print(f"\nPlayer: {player_name}")
        print(f"Seasons in dataset: {', '.join(seasons)}")
        print(f"Clubs: {', '.join(clubs)}")

        # Display teammate counts by season
        print("\nTeammate counts by season:")
        for season in seasons:
            if (
                season in self.season_networks
                and player_name in self.season_networks[season]
            ):
                season_teammates = self.season_networks[season].degree(player_name)
                print(f"  {season}: {season_teammates} teammates")

        # Display total unique teammates
        if player_name in self.G:
            total_teammates = self.G.degree(player_name)
            print(f"Total unique teammates across all seasons: {total_teammates}")

            # Show notable teammates
            teammates = list(self.G.neighbors(player_name))
            if teammates:
                teammates_sorted = sorted(
                    teammates,
                    key=lambda t: self.G[player_name][t]["weight"],
                    reverse=True,
                )
                print("\nTop teammates:")
                for t in teammates_sorted[:5]:
                    seasons = self.G[player_name][t].get("history", [])
                    season_text = ", ".join(
                        [f"{club} ({season})" for club, season in seasons]
                    )
                    print(f"  - {t}: {season_text}")


def display_menu():
    print("\n" + "=" * 50)
    print("PREMIER LEAGUE NETWORK ANALYZER".center(50))
    print("=" * 50)
    print("1. Find connection between two players")
    print("2. Show most connected players (all seasons)")
    print("3. Show most connected players (by season)")
    print("4. Visualize network for a player")
    print("5. Get player information")
    print("6. Visualize club networks")
    print("7. Visualize player club movements")
    print("8. Exit")
    print("-" * 50)


def get_player_input(network, prompt):
    while True:
        name = input(prompt)
        if not name:
            return None

        if name in network.all_players:
            return name

        suggestions = network.get_player_suggestions(name)
        if not suggestions:
            print("No matching players found. Try again.")
            continue

        print("\nDid you mean:")
        for i, player in enumerate(suggestions):
            print(f"{i + 1}. {player}")
        print(f"{len(suggestions) + 1}. None of these / Try again")

        try:
            choice = int(input("Enter number: "))
            if 1 <= choice <= len(suggestions):
                return suggestions[choice - 1]
        except ValueError:
            pass

        print("Let's try again.")


def main():
    print("\nWelcome to the Premier League Network Analyzer!")

    # Initialize and load dataset
    network = PremierLeagueNetwork("Combined_PremierLeague.csv")
    if not network.load_dataset():
        print("Failed to load dataset. Exiting.")
        return

    # Build the network
    print("\nBuilding the player network. This might take a moment...")
    if not network.build_teammate_network():
        print("Failed to build network. Exiting.")
        return

    # Main program loop
    while True:
        display_menu()
        choice = input("Enter your choice (1-8): ")

        if choice == "1":
            player1 = get_player_input(network, "Enter first player name: ")
            if player1:
                player2 = get_player_input(network, "Enter second player name: ")
                if player2:
                    network.find_shortest_path(player1, player2)
                    input("\nPress Enter to continue...")

        elif choice == "2":
            try:
                n = int(input("How many top players to show? (default: 10): ") or 10)
                network.find_most_connected_players(top_n=n)
                input("\nPress Enter to continue...")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "3":
            seasons = sorted(network.players_df[network.season_col].unique())
            print("Available seasons:")
            for i, season in enumerate(seasons):
                print(f"{i + 1}. {season}")

            try:
                season_idx = int(input("Select season (number): ")) - 1
                if 0 <= season_idx < len(seasons):
                    n = int(
                        input("How many top players to show? (default: 10): ") or 10
                    )
                    network.find_most_connected_players(
                        top_n=n, by_season=seasons[season_idx]
                    )
                else:
                    print("Invalid season selection.")
            except ValueError:
                print("Please enter valid numbers.")
            input("\nPress Enter to continue...")

        elif choice == "4":
            player = get_player_input(
                network, "Enter player name (or press Enter for full network): "
            )
            try:
                depth = int(input("Connection depth to show (1-3, default: 1): ") or 1)
                depth = max(1, min(3, depth))  # Limit between 1-3
                save = input("Save visualization? (y/n, default: n): ").lower() == "y"
                save_path = "network_viz.png" if save else None
                network.visualize_network(player, depth, save_path)
            except ValueError:
                print("Please enter a valid depth number.")

        elif choice == "5":
            player = get_player_input(network, "Enter player name: ")
            if player:
                network.get_player_stats(player)
                input("\nPress Enter to continue...")

        elif choice == "6":
            print(
                "\nClub Network Options: 1. Transfers between clubs  2. Shared players between clubs"
            )
            network_choice = input("Select visualization type (1-2): ")
            try:
                n = int(input("Number of top clubs to show (default: 20): ") or 20)
                save = input("Save visualization? (y/n, default: n): ").lower() == "y"
                save_path = "club_network_viz.png" if save else None

                if network_choice == "1":
                    network.visualize_club_network(
                        "transfers", top_n=n, save_path=save_path
                    )
                elif network_choice == "2":
                    network.visualize_club_network(
                        "shared_players", top_n=n, save_path=save_path
                    )
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Please enter valid numbers.")

        elif choice == "7":
            print(
                "\nPlayer Movement Options: 1. Specific player  2. Top players with most club changes"
            )
            viz_choice = input("Select option (1-2): ")

            if viz_choice == "1":
                player = get_player_input(network, "Enter player name: ")
                if player:
                    save = (
                        input("Save visualization? (y/n, default: n): ").lower() == "y"
                    )
                    save_path = f"{player}_movements.png" if save else None
                    network.visualize_player_club_movements(
                        player=player, save_path=save_path
                    )
            elif viz_choice == "2":
                try:
                    n = int(input("Number of top players to show (default: 5): ") or 5)
                    save = (
                        input("Save visualization? (y/n, default: n): ").lower() == "y"
                    )
                    save_path = "player_movements.png" if save else None
                    network.visualize_player_club_movements(
                        top_n=n, save_path=save_path
                    )
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("Invalid choice.")

        elif choice == "8":
            print("Thank you for using the Premier League Network Analyzer!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
