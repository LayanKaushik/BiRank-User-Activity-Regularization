# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import networkx as nx
import scipy.sparse as spa
from networkx.algorithms import bipartite
import scipy.io
import random
from scipy import stats
import matplotlib.pyplot as plt


def plot_bipartite_graph(B, title="Bipartite Graph", figsize=(72, 48), label_font_size=20, title_font_size=50):
    """
    Plots a bipartite graph.

    Parameters:
    B (networkx.Graph): The bipartite graph to be plotted.
    title (str): Title of the plot.
    figsize (tuple): Figure size as (width, height).
    label_font_size (int): Font size for node labels.
    title_font_size (int): Font size for the plot title.
    """
    plt.figure(figsize=figsize)  # Set the figure size

    # Separate the graph nodes into two groups
    top_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    bottom_nodes = set(B) - top_nodes

    # Calculate node sizes based on degree
    top_node_sizes = [B.degree(n) * 100 for n in top_nodes]
    bottom_node_sizes = [B.degree(n) * 100 for n in bottom_nodes]

    # Create a layout for the nodes
    pos = nx.bipartite_layout(B, top_nodes)

    # Draw the nodes
    nx.draw_networkx_nodes(B, pos, nodelist=top_nodes, node_size=top_node_sizes, node_color='skyblue')
    nx.draw_networkx_nodes(B, pos, nodelist=bottom_nodes, node_size=bottom_node_sizes, node_color='lightgreen')

    # Draw the edges
    nx.draw_networkx_edges(B, pos, edge_color='gray', alpha=0.5)

    # Add labels to the nodes
    nx.draw_networkx_labels(B, pos, font_size=label_font_size)

    # Add a title to the plot
    plt.title(title, fontsize=title_font_size)
    plt.axis('off')  # Hide the axes
    plt.show()

def plot_user_degree_distribution(B, users, figsize=(30, 10), bar_color='skyblue', edge_color='black', rotation=45):
    """
    Plots the degree distribution of user nodes in a bipartite graph.

    Parameters:
    B (networkx.Graph): The bipartite graph.
    users (list): List of user nodes in the graph.
    figsize (tuple): Size of the figure (width, height).
    bar_color (str): Color of the bars in the plot.
    edge_color (str): Color of the bar edges in the plot.
    rotation (int): Degree of rotation for the x-axis labels.
    """
    # Calculate degrees for user nodes
    user_degrees = [(user, B.degree(user)) for user in users]

    # Sort the users by their degrees in descending order
    user_degrees.sort(key=lambda x: x[1], reverse=True)

    # Handle the case where there are no users to plot
    if not user_degrees:
        print("No user degrees to plot.")
        return

    # Extract users and their corresponding degrees
    sorted_users, sorted_degrees = zip(*user_degrees)
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.bar(sorted_users, sorted_degrees, color=bar_color, edgecolor=edge_color)
    
    # Setting plot labels and title
    plt.xlabel('Users')
    plt.ylabel('Degree')
    plt.title('Degree Distribution of Users', fontsize=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=rotation)
    
    # Display the plot
    plt.show()

def plot_user_scores(users, user_scores, high_activity_users, title):
    """
    Plots user scores, highlighting high-activity users in a different color.

    Parameters:
    - users (list): List of users.
    - user_scores (array): Array of user scores.
    - high_activity_users (list): List of high activity user identifiers.
    - title (str): Title for the plot.
    """

    # Rank users based on their scores
    user_ranking = np.argsort(-user_scores)
    
    # Set color for each user based on their activity level
    colors = ['red' if users[i] in high_activity_users else 'blue' for i in user_ranking]
    
    # Create and display the plot
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(user_scores)), user_scores[user_ranking], color=colors)
    plt.xlabel('Users')
    plt.ylabel('Scores')
    plt.title(title)
    plt.xticks(range(len(user_scores)), user_ranking, rotation=90)
    plt.show()

def plot_product_rank_changes_ordered(products, product_scores, product_scores_reg, title):
    """
    Plots changes in product rankings, with positive changes on the left and 
    negative changes on the right.

    Parameters:
    - products (list): List of products.
    - product_scores (array): Array of original product scores.
    - product_scores_reg (array): Array of regularized product scores.
    - title (str): Title for the plot.
    """

    # Calculate rank changes for each product
    product_ranking = np.argsort(-product_scores)
    product_ranking_reg = np.argsort(-product_scores_reg)
    rank_changes = [np.where(product_ranking_reg == idx)[0][0] - np.where(product_ranking == idx)[0][0]
                    for idx in range(len(product_scores))]

    # Sort products by rank change values
    products_sorted_by_change = sorted(zip(products, rank_changes), key=lambda x: x[1])
    sorted_products, sorted_changes = zip(*products_sorted_by_change)

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(sorted_changes)), sorted_changes, color='blue')
    plt.xlabel('Products')
    plt.ylabel('Rank Change')
    plt.title(title)
    plt.xticks(range(len(sorted_changes)), sorted_products, rotation=90)
    plt.axhline(0, color='grey', lw=2)  # Line for no change
    plt.show()

def calculate_disparity(user_scores, user_activity, threshold=0.04):
    """
    Calculates the average ranks for items interacted by highly active and less active users.

    Parameters:
    user_scores (list or numpy array): Scores of the users.
    user_activity (list or numpy array): Activity levels of the users.
    threshold (float): Threshold to determine high activity users.

    Returns:
    tuple: Average rank of high activity users, average rank of low activity users.
    """
    high_activity_avg_rank = np.mean([score for score, activity in zip(user_scores, user_activity) if activity > threshold])
    low_activity_avg_rank = np.mean([score for score, activity in zip(user_scores, user_activity) if activity <= threshold])

    return high_activity_avg_rank, low_activity_avg_rank

def calculate_gini_coefficient(scores):
    """
    Calculates the Gini coefficient for a set of scores.

    Parameters:
    scores (list or numpy array): Scores for which to calculate the Gini coefficient.

    Returns:
    float: Gini coefficient.
    """
    scores = np.sort(scores)
    n = len(scores)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * scores)) / (n * np.sum(scores))

def rank_correlation(original_scores, regularized_scores):
    """
    Calculates Spearman's rank correlation between original and regularized scores.

    Parameters:
    original_scores (list or numpy array): Original scores before regularization.
    regularized_scores (list or numpy array): Scores after regularization.

    Returns:
    float: Spearman's rank correlation coefficient.
    """
    return stats.spearmanr(original_scores, regularized_scores).correlation

def top_n_analysis(original_ranking, regularized_ranking, n=10):
    """
    Identifies the top N items in both original and regularized rankings.

    Parameters:
    original_ranking (list or numpy array): Original ranking of items.
    regularized_ranking (list or numpy array): Regularized ranking of items.
    n (int): Number of top items to consider.

    Returns:
    tuple: Sets of top N items in original and regularized rankings.
    """
    original_top_n = set(original_ranking[:n])
    regularized_top_n = set(regularized_ranking[:n])

    return original_top_n, regularized_top_n

def overlap_in_top_n(original_ranking, regularized_ranking, n=10):
    """
    Calculates the overlap in top N items between original and regularized rankings.

    Parameters:
    original_ranking (list or numpy array): Original ranking of items.
    regularized_ranking (list or numpy array): Regularized ranking of items.
    n (int): Number of top items to consider.

    Returns:
    float: Proportion of overlap in the top N items.
    """
    original_top_n = set(original_ranking[:n])
    regularized_top_n = set(regularized_ranking[:n])
    overlap = original_top_n.intersection(regularized_top_n)
    return len(overlap) / n

def interaction_analysis(user_activity, user_scores):
    """
    Plots the interaction between user activity and user scores.
    
    Parameters:
    user_activity (list or numpy array): Activity levels of the users.
    user_scores (list or numpy array): Scores of the users.
    """
    plt.figure(figsize=(12, 8))  # Set the figure size
    plt.scatter(user_activity, user_scores, color='blue', edgecolor='black', alpha=0.7, s=50)  # Plot scatter
    plt.xlabel('User Activity (Degree)', fontsize=14)  # X-axis label
    plt.ylabel('User Score (BiRank)', fontsize=14)  # Y-axis label
    plt.title('Interaction Analysis: User Activity vs. Score', fontsize=16)  # Title
    plt.grid(True)  # Add a grid
    plt.show()

def fairness_metric(high_activity_avg_rank, low_activity_avg_rank, gini_coefficient, weight=0.5, epsilon=1e-6):
    """
    Calculates a fairness metric combining disparity ratio and Gini coefficient.
    Parameters:
    high_activity_avg_rank (float): Average rank of high-activity users.
    low_activity_avg_rank (float): Average rank of low-activity users.
    gini_coefficient (float): Gini coefficient of the user scores.
    weight (float): Weight factor for balancing disparity and Gini coefficient.
    epsilon (float): Small constant to avoid division by zero.
    
    Returns:
    float: Fairness metric value.
    """
    disparity_ratio = high_activity_avg_rank / (low_activity_avg_rank + epsilon)
    fairness_metric_value = 1 / ((1 - weight) / (abs(disparity_ratio - 1) + epsilon) + weight / (gini_coefficient + epsilon))
    
    return fairness_metric_value

def generate_random_bipartite_graph(num_users, num_products, density=0.1):
    """
    Generates a random bipartite graph.
    
    Parameters:
    num_users (int): Number of user nodes.
    num_products (int): Number of product nodes.
    density (float): Probability of creating an edge between user and product.
    
    Returns:
    tuple: Tuple containing the graph, user nodes, and product nodes.
    """
    graph = nx.Graph()
    user_nodes = [f'U{i}' for i in range(num_users)]
    product_nodes = [f'P{i}' for i in range(num_products)]
    
    # Adding user and product nodes to the graph
    graph.add_nodes_from(user_nodes, bipartite=0)
    graph.add_nodes_from(product_nodes, bipartite=1)
    
    # Adding edges based on the specified density
    for user in user_nodes:
        for product in product_nodes:
            if np.random.rand() <= density:
                graph.add_edge(user, product)

    return graph, user_nodes, product_nodes

def generate_power_law_bipartite_graph(num_users, num_products, user_exponent=2.5, product_degree_min=1, product_degree_max=10):
    """
    Generates a bipartite graph based on power law distribution for user nodes 
    and a uniform distribution for product nodes.

    Parameters:
    - num_users (int): Number of user nodes.
    - num_products (int): Number of product nodes.
    - user_exponent (float): Exponent for the power-law distribution of users.
    - product_degree_min (int): Minimum degree for products.
    - product_degree_max (int): Maximum degree for products.

    Returns:
    - B (NetworkX Graph): The bipartite graph.
    - users (list): List of user node labels.
    - products (list): List of product node labels.
    """

    # Generate degrees for users and products
    user_degrees = np.random.zipf(a=user_exponent, size=num_users)
    product_degrees = np.random.randint(product_degree_min, product_degree_max + 1, num_products)
    # Balance the total degree of users and products

    while sum(user_degrees) != sum(product_degrees):
        difference = sum(user_degrees) - sum(product_degrees)
        random_indices = np.random.choice(num_products, abs(difference), replace=True)
        if difference > 0:
            for i in random_indices:
                product_degrees[i] += 1
        else:
            for i in random_indices:
                product_degrees[i] = max(product_degree_min, product_degrees[i] - 1)
    
    # Construct the bipartite graph using the configuration model
    B = nx.bipartite.configuration_model(user_degrees, product_degrees)
    B = nx.Graph(B)  # Convert to a simple graph to eliminate parallel edges and self-loops
    B.remove_edges_from(nx.selfloop_edges(B))
    
    # Relabel nodes for clarity
    label_mapping = {i: f'U{i}' for i in range(num_users)}
    label_mapping.update({i + num_users: f'P{i}' for i in range(num_products)})
    B = nx.relabel_nodes(B, label_mapping)
    
    # Create lists of user and product node labels
    users = [f'U{i}' for i in range(num_users)]
    products = [f'P{i}' for i in range(num_products)]

    return B, users, products
