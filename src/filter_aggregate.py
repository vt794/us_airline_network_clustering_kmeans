
import pandas as pd
import os
import numpy as np
from docopt import docopt

pd.reset_option('display.float_format')
pd.options.display.float_format = '{:.0f}'.format


class Filter():

    def __init__(self, args):
        """
        Defines class variables from arguments passed by docopt. 

        Paramters
        ---------
        state : str (optional)
            2-letter state abbreviation, capitalized, multiples split on `,`
        airport : str
            3-letter state abbreviation, capitalized, multiples split on `,`
        RPCarrier : str
            3-char reporting carrier designator, capitalized, multiples split on `,`
        roundtrip : bool
            OD itin roundtrip flag
        distance_min : int
            Maximum combined distance of coupons across OD itin
        distance_max : int
            Maximum combined distance of coupons across OD itin
        n_clusters : int
            User specified number of Kmeanas clusters
        save_dir : str
            Path to data directory, defaults to `./data/`
        """

        self.args = args

        self.save_dir = str(self.args['--save_dir'])

        if args['--state'] != "":
            if "," in args['--state']:
                self.state = str.split(
                    args['--state'], ",")
            else:
                self.state = str(args['--state'])

        if args['--airport'] != "":
            if "," in args['--airport']:
                self.airport = str.split(
                    args['--airport'], ",")
            else:
                self.airport = str(args['--airport'])

        if args['--rpcarrier'] != "":
            if "," in args['--rpcarrier']:
                self.RPCarrier = str.split(
                    args['--rpcarrier'], ",")
            else:
                self.RPCarrier = str(args['--rpcarrier'])

        if args["--distance_min"] != "":
            self.distance_min = int(args["--distance_min"])

        if args["--distance_max"] != "":
            self.distance_max = int(args["--distance_max"])

        if args["--roundtrip"] != "":
            self.roundtrip = bool(args["--roundtrip"])

    def read_tickets(self):

        self.tickets_pp = pd.read_csv(
            self.save_dir + "processed/tickets_pp.csv")

    def filter_subset(self):
        """
        Filters ticket itins based on user specified parameters.
        """

        tickets_pp = self.tickets_pp

        if self.args['--state'] != "":
            if type(self.state) is list:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["OriginState_ndir"].isin(self.state) |
                    tickets_pp["DestState_ndir"].isin(self.state)]
            elif type(self.state) is str:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["State_ndir"].str.contains(self.state)]
            else:
                raise ValueError(
                    "self.state dtype may only be str or list")

        if self.args['--airport'] != "":
            if type(self.airport) is list:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["Origin_ndir"].isin(self.airport) |
                    tickets_pp["Dest_ndir"].isin(self.airport)]
            elif type(self.airport) is str:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["OD_ndir"].str.contains(self.airport)]
            else:
                raise ValueError(
                    "self.airport dtype may only be str or list")

        if self.args['--rpcarrier'] != "":
            if type(self.RPCarrier) is list:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["RPCarrier"].isin(self.RPCarrier)]
            elif type(self.RPCarrier) is str:
                tickets_pp = tickets_pp.loc[
                    tickets_pp["RPCarrier"].str.contains(self.RPCarrier)]
            else:
                raise ValueError(
                    "self.airport dtype may only be str or list")

        if self.args["--distance_min"] != "":
            tickets_pp = tickets_pp.loc[
                tickets_pp["Distance"] > self.distance_min]

        if self.args["--distance_max"] != "":
            tickets_pp = tickets_pp.loc[
                tickets_pp["Distance"] < self.distance_max]

        if self.args["--roundtrip"] != "":
            tickets_pp = tickets_pp.loc[
                tickets_pp["RoundTrip"] == self.roundtrip]

        self.tickets_pp = tickets_pp

    def save_tickets(self):
        """
        Saves preprocessed and filtered ticket table to csv file format.
        """

        print("saving filtered tickets locally")

        save_dir = self.save_dir + "processed/"

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.tickets_pp.to_csv(save_dir + "tickets_pp_filtered.csv",
                               index=False,
                               encoding="utf-8", escapechar='|')


class Aggregate():

    def __init__(self, args):
        """
        Defines class variables from arguments passed by docopt. 

        Paramters
        ---------
        save_dir : str
            Path to data directory, defaults to `./data/`
        """

        self.args = args
        self.save_dir = str(self.args['--save_dir'])

    def read_tickets(self):

        self.tickets_pp = pd.read_csv(
            self.save_dir + "processed/tickets_pp_filtered.csv")

        self.unique_ods_ndir = self.tickets_pp["OD_ndir"].unique()

    def build_prop_Stops(self):
        """
        Builds OD-indexed proportion of non-stop, one-stop, and multi-stop 
        comprising itins, respectively.
        """

        tickets_pp = self.tickets_pp

        tickets_pp_nonstop = tickets_pp.loc[
            ((tickets_pp["RoundTrip"] == 0) & (tickets_pp["Coupons"] == 1) |
             (tickets_pp["RoundTrip"] == 1) & (tickets_pp["Coupons"] == 2))]

        tickets_pp_onestop = tickets_pp.loc[
            ((tickets_pp["RoundTrip"] == 0) & (tickets_pp["Coupons"] == 2)) |
            (tickets_pp["RoundTrip"] == 1) & (tickets_pp["Coupons"].isin([3, 4]))]

        tickets_pp_multistop = tickets_pp.loc[
            ((tickets_pp["RoundTrip"] == 0) & (tickets_pp["Coupons"] > 2)) |
            (tickets_pp["RoundTrip"] == 1) & (tickets_pp["Coupons"] > 4)]

        d = {"non_stop": pd.Series(tickets_pp_nonstop.groupby("OD_ndir")["ItinID"].count()),
             "one_stop": pd.Series(tickets_pp_onestop.groupby("OD_ndir")["ItinID"].count()),
             "multi_stop": pd.Series(tickets_pp_multistop.groupby("OD_ndir")["ItinID"].count()),
             "total": pd.Series(tickets_pp.groupby("OD_ndir")["ItinID"].count())}

        prop_Stops = pd.DataFrame(data=d).fillna(0)
        prop_Stops["prop_nonstop"] = prop_Stops["non_stop"] / \
            prop_Stops["total"]
        prop_Stops["prop_onestop"] = prop_Stops["one_stop"] / \
            prop_Stops["total"]
        prop_Stops["prop_multistop"] = prop_Stops["multi_stop"] / \
            prop_Stops["total"]

        prop_Stops = prop_Stops[["prop_nonstop",
                                 "prop_onestop", "prop_multistop"]]

        assert prop_Stops.shape[0] == len(self.unique_ods_ndir)

        return prop_Stops

    def build_share_top_carrier_per_od(self):
        """
        Builds OD-indexed market share proportion of top carrier, along 
        with top carrier designator (for subsequent one-hot encoding). 
        """

        tickets_pp = self.tickets_pp

        count_carrier_per_od = pd.DataFrame(
            tickets_pp.groupby("OD_ndir")["RPCarrier"].count())
        count_carrier_per_od = count_carrier_per_od.rename(
            {"RPCarrier": "ItinID_ToT"}, axis=1)
        count_carrier_per_od = count_carrier_per_od.reset_index()

        all_carriers_per_od = pd.DataFrame(tickets_pp.groupby("OD_ndir")[
                                           "RPCarrier"].value_counts())
        all_carriers_per_od = all_carriers_per_od.rename(
            {"RPCarrier": "ItinID_TopRPCarrier"}, axis=1)
        all_carriers_per_od = all_carriers_per_od.reset_index()
        all_carriers_per_od = all_carriers_per_od.sort_values(
            by=['OD_ndir', 'ItinID_TopRPCarrier'])

        top_carrier_per_od = all_carriers_per_od.drop_duplicates("OD_ndir")

        share_top_carrier_per_od = pd.merge(
            top_carrier_per_od, count_carrier_per_od, on="OD_ndir")
        share_top_carrier_per_od = share_top_carrier_per_od.rename(
            {"RPCarrier": "TopCarrier"}, axis=1)
        share_top_carrier_per_od["share_TopRPCarrier"] = (
            share_top_carrier_per_od["ItinID_TopRPCarrier"] /
            share_top_carrier_per_od["ItinID_ToT"])

        share_top_carrier_per_od = share_top_carrier_per_od[["OD_ndir",
                                                             "TopCarrier",
                                                            "share_TopRPCarrier"]]

        share_top_carrier_per_od = share_top_carrier_per_od.set_index(
            "OD_ndir")

        assert share_top_carrier_per_od.shape[0] == len(self.unique_ods_ndir)

        return share_top_carrier_per_od

    def build_prop_FareClass(self):
        """
        Builds OD-indexed proportion of coach-only, business-only, first-only, 
        mixed, and unknown comprising itins, respectively.
        """

        tickets_pp = self.tickets_pp

        tickets_pp_Coach = tickets_pp.loc[
            ((tickets_pp["FareClass_Coach_Only"] == True))]
        tickets_pp_First = tickets_pp.loc[
            ((tickets_pp["FareClass_First_Only"] == True))]
        tickets_pp_Business = tickets_pp.loc[
            ((tickets_pp["FareClass_Business_Only"] == True))]
        tickets_pp_Unknown = tickets_pp.loc[
            ((tickets_pp["FareClass_Unknown_Only"] == True))]
        tickets_pp_Mixed = tickets_pp.loc[
            ((tickets_pp["FareClass_Mixed"] == True))]

        d = {"Coach": pd.Series(tickets_pp_Coach.groupby("OD_ndir")["ItinID"].count()),
             "First": pd.Series(tickets_pp_First.groupby("OD_ndir")["ItinID"].count()),
             "Business": pd.Series(tickets_pp_Business.groupby("OD_ndir")["ItinID"].count()),
             "Unknown": pd.Series(tickets_pp_Unknown.groupby("OD_ndir")["ItinID"].count()),
             "Mixed": pd.Series(tickets_pp_Mixed.groupby("OD_ndir")["ItinID"].count()),
             "Total": pd.Series(tickets_pp.groupby("OD_ndir")["ItinID"].count())}

        prop_FareClass = pd.DataFrame(data=d).fillna(0)
        prop_FareClass["prop_Coach"] = prop_FareClass["Coach"] / \
            prop_FareClass["Total"]
        prop_FareClass["prop_First"] = prop_FareClass["First"] / \
            prop_FareClass["Total"]
        prop_FareClass["prop_Business"] = prop_FareClass["Business"] / \
            prop_FareClass["Total"]
        prop_FareClass["prop_Unknown"] = prop_FareClass["Unknown"] / \
            prop_FareClass["Total"]
        prop_FareClass["prop_Mixed"] = prop_FareClass["Mixed"] / \
            prop_FareClass["Total"]
        prop_FareClass = prop_FareClass[[
            "prop_Coach", "prop_First", "prop_Business", "prop_Unknown", "prop_Mixed"]]

        assert (prop_FareClass["prop_Coach"] + prop_FareClass["prop_First"] +
                prop_FareClass["prop_Business"] + prop_FareClass["prop_Unknown"] +
                prop_FareClass["prop_Mixed"]).gt(0.99).all()

        assert (prop_FareClass["prop_Coach"] + prop_FareClass["prop_First"] +
                prop_FareClass["prop_Business"] + prop_FareClass["prop_Unknown"] +
                prop_FareClass["prop_Mixed"]).lt(1.01).all()

        assert prop_FareClass.shape[0] == len(self.unique_ods_ndir)

        return prop_FareClass

    def aggregate(self):
        """
        Aggregates ticket-indexed features to OD-indexed observations.
        """

        tickets_pp = self.tickets_pp

        unique_ods_ndir = pd.Series(tickets_pp["OD_ndir"].unique())

        feats_prop_FareClass = self.build_prop_FareClass()
        feats_prop_TopCarrier = self.build_share_top_carrier_per_od()
        feats_prop_Stops = self.build_prop_Stops()

        feats_mean_FarePerMile = tickets_pp.groupby(
            ["OD_ndir"])["FarePerMile"].median()
        feats_mean_Distance = tickets_pp.groupby(["OD_ndir"])[
            "Distance"].mean()
        feats_prop_RoundTrip = tickets_pp.groupby(
            ["OD_ndir"])["RoundTrip"].mean()
        feats_prop_OnLine = tickets_pp.groupby(["OD_ndir"])["OnLine"].mean()
        feats_mean_TkOpDistPropMatch = tickets_pp.groupby(
            ["OD_ndir"])["TkOpDistPropMatch"].mean()

        OD_ndir = pd.DataFrame(index=unique_ods_ndir)

        OD_ndir = pd.concat([
            OD_ndir,
            feats_prop_FareClass,
            feats_prop_TopCarrier,
            feats_prop_Stops,
            feats_mean_FarePerMile,
            feats_mean_Distance,
            feats_prop_RoundTrip,
            feats_prop_OnLine,
            feats_mean_TkOpDistPropMatch
        ], axis=1)

        OD_ndir = pd.concat([OD_ndir, pd.get_dummies(
            OD_ndir["TopCarrier"],
            prefix="TopCarrier_MktShare")], axis=1)

        OD_ndir = OD_ndir.drop("TopCarrier", axis=1)
        OD_ndir = OD_ndir.reset_index()
        OD_ndir = OD_ndir.rename({"index": "OD_ndir"}, axis=1)

        self.OD_ndir = OD_ndir

    def save_od_ndir(self):
        """
        Saves OD-indexed ticketing data to csv file format.
        """

        print("saving OD_ndir locally")

        save_dir = self.save_dir + "processed/"

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.OD_ndir.to_csv(save_dir + "od_ndir.csv",
                            index=False,
                            encoding="utf-8", escapechar='|')


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    filter = Filter(args)
    filter.read_tickets()
    filter.filter_subset()
    filter.save_tickets()

    aggregate = Aggregate(args)
    aggregate.read_tickets()
    aggregate.aggregate()
    aggregate.save_od_ndir()
