from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import random
from typing import Optional, List, Tuple, Set
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score, BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

class BNFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A class for generating new features based on Bayesian Network inference.

    This class constructs a Bayesian Network from input data and uses it to generate
    new features based on probabilistic inference.

    Attributes:
        known_structure (Optional[List[Tuple[str, str]]]): A list of edges representing
            the known structure of the Bayesian Network.
        bn (Optional[BayesianNetwork]): The constructed Bayesian Network.
        variables (Optional[List[str]]): List of variable names in the dataset.
        num_classes (Optional[int]): Number of unique classes in the target variable.
    """

    def __init__(self, known_structure: Optional[List[Tuple[str, str]]] = None):
        """
        Initializes the BNFeatureGenerator.

        Args:
            known_structure (Optional[List[Tuple[str, str]]]): A list of edges
                representing the known structure of the Bayesian Network.
        """
        self.known_structure = known_structure
        self.bn: Optional[BayesianNetwork] = None
        self.variables: Optional[List[str]] = None
        self.num_classes: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
            black_list: Optional[List[Tuple[str, str]]] = None) -> 'BNFeatureGenerator':
        """
        Fits the Bayesian Network to the input data.

        Arguments:
            X (pd.DataFrame): The input dataset.
            y (Optional[pd.Series]): The target variable.
            black_list (Optional[List[Tuple[str, str]]]): List of edges to be excluded
                from the Bayesian Network.

        Returns:
            self: The trained instance of BNFeatureGenerator.

        Raises:
            Exception: If there's an error fitting the BayesianNetwork with known structure.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.variables = X.columns.tolist()

        if y is not None:
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
            self.num_classes = len(y.unique())

        if self.known_structure:
            self.bn = BayesianNetwork(self.known_structure)
            try:
                # Fit the network with known structure
                self.bn.fit(X, estimator=BayesianEstimator)
            except Exception as e:
                logging.exception(f"Ошибка при обучении BayesianNetwork с известной структурой: {e}")
                raise
        else:
            # If the structure is unknown, use the constructor to build it
            constructor = self._BayesianNetworkConstructor(X, black_list)
            self.bn = constructor.construct_network()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input data by generating new features based on the trained Bayesian network.

        Arguments:
            X (pd.DataFrame): Input dataset for transformation.

        Returns:
            pd.DataFrame: A new DataFrame with generated features.

        Raises:
            KeyError: If a feature index is missing.
            Exception: For any other inference error.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        inference = VariableElimination(self.bn)

        def process_feature(feature, row):
            # Get the values of parent nodes for the current feature
            evidence = {f: row[f] for f in self.bn.get_parents(feature) if f in row.index}
            try:
                # Perform probabilistic inference
                prob = inference.query([feature], evidence=evidence)
                return prob.values[row[feature]]
            except KeyError:
                return 1 / self.num_classes if self.num_classes else 0.5
            except Exception as e:
                logging.error(f"Unexpected error in inference: {str(e)}")
                return 1 / self.num_classes if self.num_classes else 0.5

        def process_row(row):
            # Process all features for one row
            return [process_feature(feature, row) for feature in self.variables]

        # Use multithreading to speed up processing
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_row, [row for _, row in X.iterrows()]))
        # Form a DataFrame of generated features
        return pd.DataFrame(results, columns=[f'lambda_{c}' for c in self.variables])


    class _BayesianNetworkConstructor:
        """
        An internal class for constructing a Bayesian Network.

        This class implements methods for creating and modifying the structure
        of a Bayesian Network based on the input data.

        Attributes:
            data (pd.DataFrame): The input dataset.
            variables (List[str]): List of variable names in the dataset.
            repository (List[Tuple[BayesianNetwork, float]]): Repository of best networks.
            max_repository_size (int): Maximum size of the repository.
            initial_max_arcs (int): Initial maximum number of arcs allowed.
            black_list (Set[Tuple[str, str]]): Set of edges to be excluded from the network.
        """

        def __init__(self, data: pd.DataFrame, black_list: Optional[List[Tuple[str, str]]] = None):
            """
            Initializes the BayesianNetworkConstructor.

            Args:
                data (pd.DataFrame): The input dataset.
                black_list (Optional[List[Tuple[str, str]]]): List of edges to be excluded
                    from the Bayesian Network.
            """
            self.data = data
            self.variables = data.columns.tolist()
            self.repository = []
            self.max_repository_size = 10
            self.initial_max_arcs = len(self.variables) // 2
            self.black_list = set(black_list) if black_list is not None else set()

        def is_valid_edge(self, edge: Tuple[str, str]) -> bool:
            """
            Checks if an edge is valid (not in the black list).

            Args:
                edge (Tuple[str, str]): The edge to check.

            Returns:
                bool: True if the edge is valid, False otherwise.
            """
            return edge not in self.black_list

        def create_forest_structure(self):
            """
            Creates an initial forest structure for the Bayesian Network.

            Returns:
                BayesianNetwork: The initial forest structure.
            """
            nodes = self.variables.copy()
            edges = []
            used_nodes = {nodes.pop(0)}
            while nodes:
                child = nodes.pop(0)
                valid_parents = [p for p in used_nodes
                                 if self.is_valid_edge((p, child))]
                if valid_parents:
                    parent = random.Random(42).choice(valid_parents)
                    edges.append((parent, child))
                    used_nodes.add(child)
                else:
                    used_nodes.add(child)

            return BayesianNetwork(edges)

        def calculate_network_score(self, network: BayesianNetwork, sample_data: pd.DataFrame) -> float:
            """
            Calculates the score of a given Bayesian Network.

            Args:
                network (BayesianNetwork): The Bayesian Network to score.
                sample_data (pd.DataFrame): A sample of the data to use for scoring.

            Returns:
                float: The calculated score of the network.
            """
            score = BicScore(data=sample_data)
            total_score = 0
            for node in network.nodes():
                parents = list(network.get_parents(node))
                # Add a penalty for model complexity
                penalty = len(parents) * 0.1
                total_score += score.local_score(node, parents) - penalty
            return total_score

        def modify_network(self, network, max_arcs: int) -> BayesianNetwork:
            """
            Modifies the given Bayesian Network by adding, deleting, or reversing edges.

            Args:
                network (BayesianNetwork): The Bayesian Network to modify.
                max_arcs (int): The maximum number of arcs allowed in the network.

            Returns:
                BayesianNetwork: The modified Bayesian Network.

            Raises:
                ValueError: If the modification results in an invalid network structure.
                Exception: For any unexpected errors during modification.
            """
            edges = list(network.edges())
            if not edges and random.choice(['add', 'delete', 'reverse']) != 'add':
                # If no edges exist, we can only add
                return network
            # Choose operation to perform
            operation = random.choice(['add', 'delete', 'reverse'])
            new_edges = edges.copy()
            try:
                if operation == 'add' and len(edges) < max_arcs:
                    attempts = 0
                    while attempts < 10:
                        node1, node2 = random.sample(self.variables, 2)
                        new_edge = (node1, node2)
                        reverse_edge = (node2, node1)
                        # Check that the edge doesn't exist and isn't blacklisted
                        if (new_edge not in edges and reverse_edge not in edges
                                and self.is_valid_edge(new_edge)):
                            new_edges.append(new_edge)
                            break
                        attempts += 1
                elif operation == 'delete' and edges:
                    edge = random.choice(edges)
                    new_edges.remove(edge)
                elif operation == 'reverse' and edges:
                    edge = random.choice(edges)
                    reverse_edge = (edge[1], edge[0])
                    if self.is_valid_edge(reverse_edge):
                        new_edges.remove(edge)
                        new_edges.append(reverse_edge)
                # Validate the new network
                new_network = BayesianNetwork(new_edges)
                new_network.check_model()
                return new_network
            except ValueError as ve:
                # For cycles or invalid edges
                logging.exception(f"Invalid edge operation: {ve}")
            except Exception as e:
                # For unexpected errors
                logging.exception(f"Error modifying network: {e}")
            return network

        def construct_network(self, iterations: int = 100) -> BayesianNetwork:
            """
            Constructs the Bayesian Network through an iterative process.

            Args:
                iterations (int): The number of iterations for network construction.

            Returns:
                 BayesianNetwork: The constructed Bayesian Network.
            """
            current_network = self.create_forest_structure()
            sample_size = max(len(self.data) // 10, 2)
            max_arcs = self.initial_max_arcs
            # Start with a forest structure
            current_score = self.calculate_network_score(current_network,
                                                         self.data.sample(n=sample_size))

            for i in range(iterations):
                # Increase sample size and maximum number of arcs every quarter of iterations
                if i % (iterations // 4) == 0:
                    sample_size = min(sample_size * 2, len(self.data))
                    max_arcs = min(max_arcs + 2, len(self.variables) * 2)

                sample_data = self.data.sample(n=sample_size)
                new_network = self.modify_network(current_network, max_arcs)
                new_score = self.calculate_network_score(new_network, sample_data)

                if new_score > current_score:
                    # If the new network is better, update the current one and add to the repository
                    current_network = new_network
                    current_score = new_score
                    self.repository.append((new_network, new_score))
                    self.repository.sort(key=lambda x: x[1], reverse=True)
                    self.repository = self.repository[:self.max_repository_size]
                elif self.repository:
                    # If the new network is not better, choose a network from the repository
                    current_network, current_score = random.choice(self.repository)
                    current_network = self.modify_network(current_network, max_arcs)

            # Fit the final network on all data
            current_network.fit(self.data, estimator=BayesianEstimator)
            return current_network
