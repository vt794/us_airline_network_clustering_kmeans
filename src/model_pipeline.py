"""
Applies filtering to ticket-indexed data as specified by the user, aggregates ticketing 
data at a non-directional OD-level, models KMeans clusters to which observations are 
assigned, and runs unit tests.

Usage:
    src/model_pipeline.py [options]
    src/model_pipeline.py (-h | --help)
    src/model_pipeline.py

Options:
    --state=<state>                         State OD filter [default: None].
    --airport=<airport>                     Airport OD filter [default: None].
    --rpcarrier=<rpcarrier>                 RPCarrier filter [default: None].
    --roundtrip=<roundtrip>                 Rountrip ticket filter [default: None].
    --distance_min=<distance_min>           Minimum combined travel distance [default: None].
    --distance_max=<distance_max>           Maximum combined travel distance [default: None].
    --n_clusters=<n_clusters>               Number of clusters, overriding systematic n_cluster selection [default: None].
    --save_dir=<save_dir>                   Path to data directory [default: ./data/].
    --help                                  Prints this help prompt.
    --version                               Prints version.
    
    
"""


from filter_aggregate import Filter, Aggregate
from modeling import Clustering, UnitTest
from docopt import docopt


class ModelPipeline():

    def __init__(self, args):
        """
        Assigns arguments passed by docopt to self.

        Parameters
        ----------
        args : docopt
            Arguments passed via docopt.
        """

        self.args = args

    def filter_aggregate(self):
        """
        Applies filtering based on specified users paramters,
        then aggregated ticket-indexed data to non-directional
        OD-indexed data.

        For more detailed documentation, see `filter_aggregate.py`.
        """

        filter = Filter(self.args)
        filter.read_tickets()
        filter.filter_subset()
        filter.save_tickets()

        aggregate = Aggregate(self.args)
        aggregate.read_tickets()
        aggregate.aggregate()
        aggregate.save_od_ndir()

    def model(self):
        """
        Trains a KMeans clustering model using on non-directional
        OD-indexed feature data, and produces per-cluster summary 
        statistics and per-observation cluster assignment table data, 
        along with the model's Silhouette, Elbow, and Interdistance 
        plots. It applies unit tests on the modeling workflow.

        For more detailed documentation, see `modeling.py`.
        """

        model = Clustering(self.args)
        model.train_model()

        test = UnitTest(self.args)
        test.test_scaling()
        test.test_model_output()


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    data_pipeline = ModelPipeline(args)
    data_pipeline.filter_aggregate()
    data_pipeline.model()
