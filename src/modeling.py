import pandas as pd
import numpy as np
import random
import os
from docopt import docopt

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, InterclusterDistance, SilhouetteVisualizer

from scipy.spatial.distance import cdist

import pickle
import unittest


class Clustering():

    def __init__(self, args):
        """
        Initializes the clustering model object, which is trained from processed data,
        tuned to optimal parameters, and which outputs labels per each row aggregation obervation.
        """

        self.seed = 2022
        self.args = args

        if self.args["--n_clusters"] != "":
            self.n_clusters = int(args["--n_clusters"])

        self.save_dir = str(self.args['--save_dir'])

        self.read_od_ndir_data()

    def read_od_ndir_data(self):
        """
        Reads in OD-indexed feature data stored as a .csv file.
        """

        self.ndir_ods = pd.read_csv("./data/processed/od_ndir.csv")

        self.input_features = [
            col for col in self.ndir_ods.columns if (col not in ["OD_ndir", "cluster_kmeans"])]

        # return args for unit tests
        return (self.ndir_ods, self.input_features)

    def feature_scaling(self, data: pd.DataFrame):
        """
        Function applies standard scaling to input features.

        Parameters
        ----------
        data : pd.DataFrame
            Feature data set to be scaled

        Returns
        -------
        data_scaled : pd.DataFrame
            Scaled feature data
        """

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(
            data_scaled, columns=data.columns).reset_index(drop=True)

        return data_scaled

    def train_model(self, n_clusters=3, verbose=True):
        """
        Function trains model given specified parameters.

        Paramters
        ---------
        model : str, optional
            Model of choice, default set to `KMeans`
        n_clusters : int, optional
            Hyper-parameter for the number of clusters, applicable when `model=KMeans`,
            default set to 3
        """

        X = self.feature_scaling(self.ndir_ods[self.input_features])

        id_feat = "OD_ndir"

        # try:
        results, summary = self.kmeans(X=X.values, data=self.ndir_ods, n_clusters=n_clusters,
                                       id_feat=id_feat, verbose=verbose)

        # return args for unit tests
        return results, summary

        # except:
        #    print("No systematic n_cluster Elbow value detected...\nSolutions: (1) adjust filtering, (2) tune preprocessing, or (3) specify n_cluster value manually.")

    def kmeans(self, X: pd.DataFrame, data: pd.DataFrame, id_feat,
               n_clusters=3, seed=2022, verbose=True):
        """
        Fits KMeans model to data and saves trained model and output label.

        A function that trains an unsupervised learning KMeans model that groups
        observations into labeled clusters, saving resulting the trained model and
        resulting output.

        Paramters
        ---------
        X : pd.DataFrame
            Scaled training data, excluding unique id
        data : pd.DataFrame
            Original unscaled complete data, to which output labels are attached
        id_feat : string
            Id column from `data`, excluded from training but key for labeling
        n_clusters : int, optional
            Required input to fit KMeans model, no default set.
        seed : int, optional
            Random state seed used for clustering fit initialization
        verbose : bool, optional
            Print progress statements, defaults to True.
        """

        # instantiates Elbow plot visualizer, passing it n_cluster range
        model = KMeans(random_state=self.seed)
        visualizer = KElbowVisualizer(model, k=(2, 18), timings=True)

        fig = plt.figure()
        visualizer = visualizer.fit(X)

        # assigns a systematically tuned n_cluster value if none is specified
        if self.args["--n_clusters"] != "":
            self.n_clusters = visualizer.elbow_value_

        # saves render of resulting Elbow plot
        if verbose:
            print("saving render of resulting Elbow plot")

        save_dir = self.save_dir + "output/"

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        visualizer.show(outpath=save_dir +
                        f"elbow_plot__{n_clusters}_clusters.png")

        # instantiates KMeans model based on specified or assigned hyper-params
        model_KMeans = KMeans(n_clusters=n_clusters, random_state=seed)

        # instantiates and computes Silhoutte Distance plot visualizer
        fig = plt.figure()
        visualizer = SilhouetteVisualizer(model_KMeans, colors='yellowbrick')

        visualizer = visualizer.fit(X)

        # renders and saves Silhouette Distance plot
        if verbose:
            print("saving render of resulting Silhouette distance plot")

        visualizer.show(
            outpath=save_dir + f"silhoutte_plot__{n_clusters}_clusters.png")

        # instantiates and computes Intercluster Distance plot visualizer
        fig = plt.figure()
        visualizer = InterclusterDistance(model_KMeans)
        visualizer = visualizer.fit(X)

        # renders and saves Intercluster Distance plot
        if verbose:
            print("saving render of resulting Intercluster Distance plot")

        visualizer.show(
            outpath=save_dir + f"intercluster_distance_plot__{n_clusters}_clusters.png")

        # fits the KMeans model object to scaled feature data
        model_KMeans.fit(X)

        # appends cluster assignment to data
        results_kmeans = data.copy()
        results_kmeans["cluster_kmeans"] = model_KMeans.labels_

        # generates cluster summary table
        summary_kmeans = pd.concat([
            results_kmeans.groupby('cluster_kmeans')["OD_ndir"].count(),
            results_kmeans.groupby('cluster_kmeans').mean()], axis=1)
        summary_kmeans = summary_kmeans.reset_index()
        summary_kmeans = summary_kmeans.rename(
            {f"{id_feat}": f"{id_feat}_count"}, axis=1)

        # saves output result and summary, along KMeans model object
        if verbose:
            print("saving model results and summary")

        results_kmeans.to_csv(
            save_dir + f"results_labeled__{n_clusters}_clusters.csv", index=False)

        summary_kmeans.to_csv(
            save_dir + f"per_cluster_summary__{n_clusters}_clusters.csv", index=False)

        with open(save_dir + f"fit_model__{n_clusters}_clusters.pkl", "wb") as files:
            pickle.dump(model_KMeans, files)

        # return results and summary for unit tests
        return results_kmeans, summary_kmeans


class UnitTest(unittest.TestCase):

    def __init__(self, args):
        """
        Initializes the clustering model object, which is trained from processed data,
        tuned to optimal parameters, and which outputs labels per each row aggregation obervation.
        """
        self.seed = 2022
        self.args = args

    def test_scaling(self):

        print("running unit test on feature scaling")

        test_object = Clustering(self.args)
        ndir_ods = test_object.read_od_ndir_data()[0]
        input_features = test_object.read_od_ndir_data()[1]
        ndir_ods_scaled = test_object.feature_scaling(
            ndir_ods[input_features])

        self.assertTrue(
            len(ndir_ods) == len(ndir_ods_scaled),
            "Lengths of original and scaled data set are not equal."
        )

        self.assertTrue(
            (ndir_ods.shape[1]-1 == test_object.feature_scaling(
                ndir_ods[input_features]).shape[1]),
            ("Numerical column counts in original and scaled data " +
             "set are not equal.")
        )

        self.test_object = test_object

    def test_model_output(self):
        """
        1. Tests that the length of cluster summary table matches `n_clusters`.

        2. Tests that `results_kmeans` has one more column than `ndir_col`, 
        which contains the cluster assignment per observation.

        """

        print("running unit test on model output")

        n_clusters_test = random.randint(2, 10)

        test_object = Clustering(self.args)
        ndir_ods = test_object.read_od_ndir_data()[0]
        results_kmeans, summary_kmeans = test_object.train_model(
            n_clusters=n_clusters_test, verbose=False)

        if self.args["--n_clusters"] != "":
            self.n_clusters = int(self.args["--n_clusters"])

        # 1
        self.assertTrue(
            len(summary_kmeans) == n_clusters_test,
            ("Length of cluster summary table is not equal to the number " +
             "of clusters specified in test.")
        )

        # 2
        self.assertTrue(
            (results_kmeans.shape[1] - 1) == ndir_ods.shape[1],
            ("Length of cluster-labeled data table is not equal to the number " +
             "of clusters specified in test.")
        )


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    model = Clustering(args)
    model.train_model()

    test = UnitTest(args)
    test.test_scaling()
    test.test_model_output()
