import numpy
from bamt.networks.hybrid_bn import HybridBN
import pandas as pd
import bamt.preprocessors as pp
from typing import Optional, List, Tuple
import logging
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from pgmpy.estimators import K2Score
import numpy as np
from bamt.preprocess.discretization import code_categories, get_nodes_type
from bamt.networks import DiscreteBN, ContinuousBN
from scipy.stats import norm

class BNFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates features based on a Bayesian Network (BN).
    """
    def __init__(self, known_structure: Optional[List[Tuple[str, str]]] = None,
                 black_list: Optional[List[Tuple[str, str]]] = []):

        self.known_structure = known_structure
        self.bn = None
        self.variables: Optional[List[str]] = None
        self.encoder = preprocessing.LabelEncoder()
        self.discretizer = preprocessing.KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy='kmeans',
            subsample=None
        )
        self.black_list = black_list
        self.preprocessor = pp.Preprocessor([('encoder', self.encoder), ('discretizer', self.discretizer)])
        self.nodes_dict = {} # Dictionary to store nodes of the Bayesian Network
        self.target_name = ''

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fits the BNFeatureGenerator to the data.

        This involves:
        1.  Adding the target variable (if provided) to the input data.
        2.  Encoding categorical columns.
        3.  Discretizing continuous columns.
        4.  Creating a Bayesian Network based on the data types.
        5.  Learning the structure of the Bayesian Network (if known_structure is not provided).
        6.  Fitting the parameters of the Bayesian Network.

        Args:
            X (pd.DataFrame): The input data.
            y (Optional[pd.Series]): The target variable. If provided, it will be added to the input data and
                treated as a node in the Bayesian Network.

        Returns:
            self: The fitted BNFeatureGenerator object.
        """
        if y is not(None):
            X = pd.concat([X, y], axis = 1).reset_index(drop=True)
            self.target_name = y.name
        cat_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        if cat_columns:
            X, cat_encoder = code_categories(X, method="label", columns=cat_columns)
        X = pd.DataFrame(X)
        discretized_data, est = self.preprocessor.apply(X)
        discretized_data = pd.DataFrame(discretized_data, columns=X.columns)
        info = self.preprocessor.info # Get information about the data types after preprocessing
        disc_columns = 0
        cont_columns = 0
        get_nodes_type = info['types'] # Get types of nodes
        print(get_nodes_type)

        for key, value in get_nodes_type.items(): # Iterate through node types and count discrete and continuous columns
            if value == 'disc' or value == 'disc_num':
                disc_columns += 1
            else:
                cont_columns += 1
        disc_columns+=len(cat_columns)

        # Based on the amount of discrete and continuous columns:
        if cont_columns == 0:
            self.bn = DiscreteBN() # Create Discrete Bayesian Network
            logging.info("Using DiscreteBN")
            self.bn.add_nodes(info)
        elif disc_columns == 0:
            self.bn = ContinuousBN(use_mixture=False) # Create Continuous Bayesian Network
            logging.info("Using ContinuousBN")
            self.bn.add_nodes(info)
        else:
            self.bn = HybridBN(has_logit=True, use_mixture=False) # Create Hybrid Bayesian Network
            logging.info("Using HybridBN")
            self.bn.add_nodes(info)

        if self.known_structure:
            params = {'init_edges': self.known_structure}
            self.bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
        else:
            params = {}
            if self.target_name:
                bl = self.black_list + self.create_black_list(X, self.target_name) # Edges to avoid
                params = {'bl_add': bl}
            self.bn.add_edges(discretized_data, scoring_function=('K2', K2Score), params=params)
        print('bn',self.bn.edges)
        self.bn.fit_parameters(X) # Fit parameters of the Bayesian Network

        for node in self.bn.nodes:
                node.node_info = node.fit_parameters(X) # Fit parameters for each node

        self.variables = list(X.columns)
        self.nodes_dict = {str(node.name): node for node in self.bn.nodes}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame `X` into a new DataFrame where each column
        represents the calculated feature based on the fitted BN.

        Args:
            X (pd.DataFrame) is the input DataFrame to transform.

        Returns:
            pd.DataFrame is a new DataFrame with lambda-features.
        """
        if not self.bn:
            logging.error(AttributeError,
                "Parameter learning wasn't done. Call fit method"
            )
            return pd.DataFrame()  # Return an empty DataFrame to avoid further errors

        # Handle categorical columns (if any)
        cat_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_columns:
            X, cat_encoder = code_categories(X, method="label", columns=cat_columns)

        results = []
        # Process each feature (column) in the row using the BN
        for _, row in X.iterrows():
            row_probs = [self.process_feature(feat, row) for feat in self.variables]
            results.append(row_probs)

        return pd.DataFrame(results, columns=['lambda_' + c for c in self.variables])


    def create_black_list(self, X: pd.DataFrame, y: Optional[str]):
        if not y:
            return []
        target_node = y
        black_list = [(target_node, (col)) for col in X.columns.to_list() if col != target_node]

        return black_list

    def val_to_str(self, val):
        """
        Converts a value to its string representation.
        """
        if isinstance(val, float) and val.is_integer():
            return str(int(val))
        elif isinstance(val, int):
            return str(val)
        elif isinstance(val, str):
            try:
                f = float(val)
                if f.is_integer():
                    return str(int(f))
                else:
                    return val
            except Exception:
                return val
        else:
            return str(val)
    def process_feature(self, feature: str, row: pd.Series):
        """
        Processes a single feature (node) in the Bayesian network for a given row of data.

        Args:
            feature (str): The name of the feature (node) being processed.
            row (pd.Series): A row from X.

        Returns:
            float: The probability or observed value depending on the node type.
        """
        try:
            node = self.nodes_dict[str(feature)]
            node_info = getattr(node, 'node_info', None)

            if node_info is None:
                raise ValueError(f"Node info is missing for node {feature}")

            parents = node.disc_parents + node.cont_parents
            print(parents)

            pvals = []
            pvals_no_cont = []
            pvals_cont = []

            # Iterate through the continuous parents
            for p in node.cont_parents:
                pvals.append(row[p])
                pvals_cont.append(row[p])

            # Iterate through the discrete parents
            for p in node.disc_parents:
                print(p, row[p])
                norm_val = self.val_to_str(row[p])
                pvals.append(norm_val)
                pvals_no_cont.append(norm_val)

            # Process discrete nodes
            if node.type == 'Discrete':
                return self._process_discrete_node(node, node_info, feature, row, pvals)
            # Process non-discrete nodes
            else:
                return self._process_non_discrete_node(node, node_info, feature, row, pvals, pvals_no_cont,
                                                       pvals_cont)

        except Exception as e:
            logging.error(f"Error processing node {feature}: {e}")
            return 0.0001

    def _process_discrete_node(self, node, node_info, feature, row, pvals):
        """
        Processes a discrete node.

        Args:
            node - the discrete node object.
            node_info - information about the node.
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.

        Returns:
            float - value of a new feature.
        """

        if str(feature) == self.target_name:
            # If the current feature is the target variable, predict its value
            obs_value = str(node.predict(node_info=node_info, pvals=pvals))
            print('obs_value', obs_value)
            return float(obs_value)
        else:
            # If the current feature is not the target variable, get the observed value from the row
            obs_value = str(int(row[feature]))
        try:
            dist = node.get_dist(node_info=node_info, pvals=pvals)
        except:
            return 0.0001
            #pvals = list(map(str, pvals))
            #dist = node.get_dist(node_info=node_info, pvals=pvals)
        print(dist)
        try:
            # Try to find the index of the observed value in the node's values
            idx = node_info["vals"].index(obs_value)
        except:
            print()
            idx = node_info["vals"].index(float(obs_value))

        return dist[idx]

    def _process_non_discrete_node(self, node, node_info, feature, row, pvals, pvals_no_cont, pvals_cont):
        """
        Processes a non-discrete node.

        Args:
            node - the discrete node object.
            node_info - information about the node.
            feature (str) - the name of the feature (node).
            row (pd.Series) - a row of data from the DataFrame.
            pvals (list) - list of parent values.

        Returns:
            float - value of a new feature.
        """
        if str(feature) == self.target_name:
            # If the current feature is the target variable, predict its value
            obs_value = float(node.predict(node_info=node_info, pvals=pvals))
            return obs_value
        else:
            # If the current feature is not the target variable, get the observed value from the row
            obs_value = float(row[feature])
        try:
            dist = node.get_dist(node_info=node_info, pvals=pvals)
        except:
            return 0.0001
            #pvals = list(map(str, pvals))
            #dist = node.get_dist(node_info=node_info, pvals=pvals)


        if isinstance(dist, tuple):
            try:
                # If the distribution is a tuple (meaning these are a mean and variance)
                if len(dist) == 2:
                    mean, variance = dist
                    sigma = np.sqrt(variance)

                    epsilon = 1
                    prob = norm.cdf(obs_value + epsilon, loc=mean, scale=sigma) - norm.cdf(obs_value - epsilon, loc=mean, scale=sigma)

                    if  numpy.isnan(prob):
                        return 0.0001
                    return prob
                else:
                    logging.warning("unknown dist for node %s: %s", feature, dist)
                    return 0.0001
            except:
                logging.warning(" dist not found for node %s:", feature)
                return 0.0001

        # If the distribution is an array (meaning we're processing Logit node)
        if isinstance(dist, np.ndarray):
            try:
                if "classes" in node_info["hybcprob"][str(pvals_no_cont)]:
                    classes = node_info["hybcprob"][str(pvals_no_cont)]["classes"]
                    if obs_value in classes:
                        idx = classes.index(obs_value)
                        prob = dist[idx]
                        return prob
                    else:
                        return 0.0001
            except:
                return 0.0001
        else:
            return 0.0001
