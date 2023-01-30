import pandas as pd
import os
import numpy as np
from docopt import docopt

pd.reset_option('display.float_format')
pd.options.display.float_format = '{:.0f}'.format


class Preprocess():

    def __init__(self, args):
        """
        Defines class variables from arguments passed by docopt. 

        Assigns int dtype to `year` and `quarter` mandatory params, and 
        assign str dtype to optional `save_dir` param. 
        """
        self.args = args

        self.year = int(self.args['--year'])
        self.quarter = int(self.args['--quarter'])
        self.save_dir = str(self.args['--save_dir'])

    def read_coupons(self, cols=None):
        """
        Reads in DB1B raw coupon data, stored in csv file format.
        """

        coupon_cols = ["ItinID", "MktID",
                       "SeqNum", "Coupons",
                       "Origin", "Dest", "Break",
                       "TkCarrier", "OpCarrier", "RPCarrier",
                       "FareClass", "Distance",
                       "OriginState", "DestState"]

        self.coupons = pd.read_csv(self.save_dir + "raw/" + f"coupons_{self.year}_{self.quarter}.csv",
                                   usecols=coupon_cols if (cols == None) else cols)

    def read_tickets(self, cols=None):
        """
        Reads in DB1B raw tickets data, stored in csv file format.
        """

        ticket_cols = ["ItinID", "Coupons",
                       "Origin", "RoundTrip", "OnLine", "DollarCred",
                       "FarePerMile", "BulkFare", "RPCarrier",
                       "ItinFare", "Distance", "ItinGeoType"]

        tickets = pd.read_csv(self.save_dir + "raw/" + f"tickets_{self.year}_{self.quarter}.csv",
                              usecols=ticket_cols if (cols == None) else cols)

        if len(tickets.ItinID.unique()) != len(tickets.ItinID):
            tickets = tickets.drop_duplicates()

        return tickets

    def read_subset_tickets(self):
        """
        Subsets ticketing data to drop perceived outliers, subject to interpretation.
        """

        # load all tickets
        tickets = self.read_tickets()
        # tickets = self.read_tickets().sample(1000)

        # subset based on `DollarCred`, `ItinGeoType`, `BulkFare`, `ItinFare`
        tickets = tickets.loc[
            (tickets["DollarCred"] == 1) &
            (tickets["ItinGeoType"] == 2) &
            (tickets["BulkFare"] == 0)].drop(
            ["DollarCred", "ItinGeoType", "BulkFare"], axis=1)
        tickets = tickets.loc[
            ((tickets["ItinFare"] > 25) & (tickets["RoundTrip"] == 0)) |
            ((tickets["ItinFare"] > 50) & (tickets["RoundTrip"] == 1))]
        tickets = tickets.loc[
            ((tickets["ItinFare"] < 2500) & (tickets["RoundTrip"] == 0)) |
            (tickets["RoundTrip"] == 1)]

        self.tickets = tickets

    def build_coupon_Flt(self):
        """
        Builds non-directionary airport OD coupon field.
        """

        coupons = self.coupons

        # create new flt_dir and flt_ndir column
        coupons["flt_dir"] = coupons[["Origin", "Dest"]].agg('-'.join, axis=1)
        coupons["flt_ndir"] = np.where(
            (coupons["Origin"] < coupons["Dest"]) == True,
            coupons[["Origin", "Dest"]].agg('-'.join, axis=1),
            coupons[["Dest", "Origin"]].agg('-'.join, axis=1))

        self.coupons = coupons

    def drop_itins_missing_FareClass(self):
        """
        Drops any itins comprising NaN in coupons `FareClass` field.
        """

        itinid_missing_FareClass = self.coupons.loc[
            self.coupons["FareClass"].isna()]["ItinID"].tolist()

        if len(itinid_missing_FareClass) > 0:
            self.coupons = self.coupons[-self.coupons["ItinID"]
                                        .isin(itinid_missing_FareClass)]
            self.tickets = self.tickets[-self.tickets["ItinID"]
                                        .isin(itinid_missing_FareClass)]
        else:
            pass

    def clean_coupon_Break(self):
        """
        Cleans `Break` data in the `coupons` data frame. 

        Replaces NaN and `X` with binary 0 and 1, respectively. Then, 
        asserts that only [0, 1] unique values remain.
        """

        self.coupons["Break"] = self.coupons["Break"].fillna(0)
        self.coupons.loc[(self.coupons["Break"] == 'X'), 'Break'] = 1
        self.coupons["Break"] = self.coupons["Break"].astype(int)

        assert self.coupons["Break"].unique().tolist() == [0, 1]

    def build_ticket_OD(self):
        """
        Retrieves systematic non-cyclical airport and state destination per each 
        ticket ItinID.

        On one-way and round-trip itins, retrieves origin airport and state 
        from coupon whose SeqNum==1. 
        On one-way itins, retrieves destination airport and state from coupon 
        whose SeqNum==Coupons.
        On round-trip itins, retrieves destination airport and state from coupon 
        whose SeqNum!=Coupons & Break==1.
        """

        coupons = self.coupons.merge(
            self.tickets[["ItinID", "RoundTrip"]], on="ItinID")

        coupon_origin = coupons.loc[coupons["SeqNum"]
                                    == 1][["ItinID", "Origin", "OriginState"]]

        coupon_destination = coupons.loc[
            ((coupons["RoundTrip"] == 0) & (coupons["SeqNum"] == coupons["Coupons"])) |
            ((coupons["RoundTrip"] == 1) & (coupons["SeqNum"] != coupons["Coupons"]) &
             (coupons["Break"] == 1))][["ItinID", "Dest", "DestState"]]

        coupon_od = pd.merge(coupon_origin, coupon_destination, on="ItinID")

        self.tickets = pd.merge(self.tickets.drop("Origin", axis=1),
                                coupon_od, on="ItinID")

    def subset_itins_on_coupon_Break(self, verbose=0):
        """
        Subsets itins to 1-break one-way and 2-break roundtrip itins. 
        """

        BreakCnt_per_id = pd.DataFrame(self.coupons.groupby("ItinID")["Break"].sum()).rename({
            "Break": "BreakCnt"}, axis=1).reset_index()

        assert (BreakCnt_per_id["BreakCnt"].isna().sum() == 0)
        assert (self.tickets["RoundTrip"].isna().sum() == 0)
        # assert (BreakCnt_per_id["ItinID"].nunique() == tickets["ItinID"].nunique())

        ItinID_break_rt_subset = BreakCnt_per_id.merge(
            self.tickets[["ItinID", "RoundTrip"]], on="ItinID")
        ItinID_break_rt_subset = ItinID_break_rt_subset.loc[
            (ItinID_break_rt_subset["BreakCnt"] == 1) & (ItinID_break_rt_subset["RoundTrip"] == 0) |
            (ItinID_break_rt_subset["BreakCnt"] == 2) & (ItinID_break_rt_subset["RoundTrip"] == 1)]

        ItinID_subset = ItinID_break_rt_subset["ItinID"].unique().tolist()
        self.coupons = self.coupons[self.coupons["ItinID"].isin(ItinID_subset)]
        self.tickets = self.tickets[self.tickets["ItinID"].isin(ItinID_subset)]

        if verbose:
            ItinID_dropcnt = BreakCnt_per_id["ItinID"].nunique(
            ) - ItinID_break_rt_subset["ItinID"].nunique()
            print(
                f"Dropping ItinIDs w/ mismatch between Coupon BreakCnt [1,2] and Ticket RoundTrip [0,1]: {ItinID_dropcnt} ItinIDs affected.")

    def build_OD_ndir(self):
        """
        Builds non-directionary combined and split OD fields per ItinID in self.tickets.
        """

        self.tickets["OD_ndir"] = np.where(
            (self.tickets["Origin"] < self.tickets["Dest"]) == True,
            self.tickets[["Origin", "Dest"]].agg('-'.join, axis=1),
            self.tickets[["Dest", "Origin"]].agg('-'.join, axis=1))

        self.tickets["Origin_ndir"] = self.tickets["OD_ndir"].str[:3]
        self.tickets["Dest_ndir"] = self.tickets["OD_ndir"].str[-3:]

    def build_State_ndir(self):
        """
        Builds non-directionary combined and split OD state fields per ItinID in self.tickets.
        """

        self.tickets["State_ndir"] = np.where(
            (self.tickets["OriginState"] < self.tickets["DestState"]) == True,
            self.tickets[["OriginState", "DestState"]].agg('-'.join, axis=1),
            self.tickets[["DestState", "OriginState"]].agg('-'.join, axis=1))

        self.tickets["OriginState_ndir"] = self.tickets["State_ndir"].str[:2]
        self.tickets["DestState_ndir"] = self.tickets["State_ndir"].str[-2:]

    def build_TkOpDistPropMatch(self):
        """
        Builds a per-itin field denoting the proportion of matching ticketing 
        and operating carrier across comprising coupons.
        """

        self.coupons["TkOpDistMatch"] = (
            self.coupons["TkCarrier"] == self.coupons["OpCarrier"]) * self.coupons["Distance"]
        self.tickets = self.tickets.merge(pd.DataFrame(self.coupons.groupby("ItinID")["TkOpDistMatch"].sum()).reset_index(),
                                          on="ItinID")
        self.tickets["TkOpDistPropMatch"] = self.tickets["TkOpDistMatch"] / \
            self.tickets["Distance"]
        self.tickets = self.tickets.drop(["TkOpDistMatch"], axis=1)

    def build_FareClass(self):
        """
        Builds a per-ticket FareClass designator from coupon FareClass data.
        """

        coupons = self.coupons

        coupons = coupons[["ItinID", "SeqNum", "FareClass"]]

        ItinID_missing_FareClass = coupons.loc[coupons["FareClass"].isna(
        )]["ItinID"].tolist()
        coupons = coupons[-coupons["ItinID"].isin(ItinID_missing_FareClass)]

        coupons["FareClass_X"] = (coupons["FareClass"] == "X")
        coupons["FareClass_Y"] = (coupons["FareClass"] == "Y")
        coupons["FareClass_D"] = (coupons["FareClass"] == "D")
        coupons["FareClass_F"] = (coupons["FareClass"] == "F")
        coupons["FareClass_C"] = (coupons["FareClass"] == "C")
        coupons["FareClass_G"] = (coupons["FareClass"] == "G")

        coupons["FareClass_Unknown"] = (coupons["FareClass"] == "U")

        coupons["FareClass_Coach"] = (
            coupons["FareClass_X"] | coupons["FareClass_Y"])
        coupons["FareClass_First"] = (
            coupons["FareClass_F"] | coupons["FareClass_G"])
        coupons["FareClass_Business"] = (
            coupons["FareClass_C"] | coupons["FareClass_D"])

        coupons_SeqCnt = pd.DataFrame(
            coupons.groupby("ItinID")["SeqNum"].count())

        coupons_ItinID_Coach = pd.DataFrame(
            coupons.groupby("ItinID")["FareClass_Coach"].sum())
        coupons_ItinID_First = pd.DataFrame(
            coupons.groupby("ItinID")["FareClass_First"].sum())
        coupons_ItinID_Business = pd.DataFrame(
            coupons.groupby("ItinID")["FareClass_Business"].sum())
        coupons_ItinID_Unknown = pd.DataFrame(
            coupons.groupby("ItinID")["FareClass_Unknown"].sum())

        merge_dfs = [coupons_SeqCnt, coupons_ItinID_Coach, coupons_ItinID_First,
                     coupons_ItinID_Business, coupons_ItinID_Unknown]

        ticket_FareClass = pd.concat(merge_dfs, axis=1, copy=False)

        ticket_FareClass["FareClass_Coach_Only"] = (
            ticket_FareClass["SeqNum"] == ticket_FareClass["FareClass_Coach"])
        ticket_FareClass["FareClass_First_Only"] = (
            ticket_FareClass["SeqNum"] == ticket_FareClass["FareClass_First"])
        ticket_FareClass["FareClass_Business_Only"] = (
            ticket_FareClass["SeqNum"] == ticket_FareClass["FareClass_Business"])
        ticket_FareClass["FareClass_Unknown_Only"] = (
            ticket_FareClass["SeqNum"] == ticket_FareClass["FareClass_Unknown"])
        ticket_FareClass["FareClass_Mixed"] = ~(
            ticket_FareClass["FareClass_Coach_Only"] |
            ticket_FareClass["FareClass_First_Only"] |
            ticket_FareClass["FareClass_Business_Only"] |
            ticket_FareClass["FareClass_Unknown_Only"])

        ticket_FareClass = ticket_FareClass[[
            "FareClass_Coach_Only", "FareClass_First_Only", "FareClass_Business_Only",
            "FareClass_Unknown_Only", "FareClass_Mixed"]].reset_index()

        assert ((ticket_FareClass["FareClass_Coach_Only"].sum() +
                ticket_FareClass["FareClass_First_Only"].sum() +
                ticket_FareClass["FareClass_Business_Only"].sum() +
                ticket_FareClass["FareClass_Mixed"].sum() +
                ticket_FareClass["FareClass_Unknown_Only"].sum()
                 ) == len(ticket_FareClass))

        self.tickets = self.tickets.merge(ticket_FareClass, on="ItinID")

    def preprocess(self):
        """
        Applies preprocessing steps in sequence.
        """

        # reads and merges raw data ahead of preprocessing
        print("reading and merging ticket and coupon data")
        self.read_subset_tickets()
        self.read_coupons()

        # preprocesses data ahead of feature engineering
        print("preprocessing ticket and coupon data")
        self.drop_itins_missing_FareClass()
        self.clean_coupon_Break()
        self.build_ticket_OD()
        self.subset_itins_on_coupon_Break()

        # engineers ticket-indexed features
        print("building ticket-indexed features")
        self.build_coupon_Flt()
        self.build_OD_ndir()
        self.build_State_ndir()
        self.build_TkOpDistPropMatch()
        self.build_FareClass()

    def save_tickets_pp(self):
        """
        Saves preprocessed ticket and coupon data to csv.
        """

        print("saving preprocessed ticket and coupon data")

        save_dir = self.save_dir + "processed/"

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.tickets.to_csv(save_dir + f"tickets_pp.csv",
                            index=False,
                            encoding="utf-8", escapechar='|')

        self.coupons.to_csv(save_dir + f"coupons_pp.csv",
                            index=False,
                            encoding="utf-8", escapechar='|')


if __name__ == "__main__":

    args = docopt(__doc__, version="0.1.0")

    preprocess = Preprocess(args)
    preprocess.preprocess()
    preprocess.save_tickets_pp()
