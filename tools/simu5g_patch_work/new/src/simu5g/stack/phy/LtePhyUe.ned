//
//                  Simu5G
//
// Copyright (C) 2019-2021 Giovanni Nardini, Giovanni Stea, Antonio Virdis et al. (University of Pisa)
// Copyright (C) 2022-2026 Giovanni Nardini, Giovanni Stea et al. (University of Pisa)
//
// This file is part of a software released under the license included in file
// "license.pdf". Please read LICENSE and README files before using it.
// The above files and the present reference are part of the software itself,
// and cannot be removed from it.
//


package simu5g.stack.phy;

//
// Extends the ~LtePhyBase module by providing specific PHY functions
// on the User Equipment (UE) side. It ensures that frames received on the air interface
// are sourced from the serving eNodeB (eNB) and provides functions to manage the handover procedure,
// including periodic measurement of channel conditions to determine when a handover should
// occur. If enabled, the selection of the serving eNB at initialization is done dynamically
// based on radio conditions.
//
simple LtePhyUe extends LtePhyBase
{
    parameters:
        @class("LtePhyUe");

        bool dynamicCellAssociation = default(false);

        string rlcUmModule = default("^.rlc.um"); // or "^.nrRlc.um"
        string pdcpModule = default("^.pdcp");
        string ip2nicModule = default("^.ip2nic");
        string feedbackGeneratorModule = default("^.dlFbGen"); // TODO or "^.nrDlFbGen"

        // if true, get minRssi from PhyPisaData
        bool minRssiDefault = default(true);

        // configurable minimum threshold RSSI for attaching to an eNB (meaningful only if minRssiDefault==false)
        double minRssi @unit("dB") = default(-99.0dB);

        bool hasCollector = default(false); // true when node has a collector module (ueCollector/nrUeCollector) for this PHY module

        // Railway/RL handover extensions. Defaults preserve the original Simu5G behavior.
        double hysteresisFactor = default(10);
        double handoverDelta @unit(s) = default(0.00001s);
        bool railwayA3TttHandover = default(false);
        double railwayA3Hysteresis @unit(dB) = default(3dB);
        double railwayA3TimeToTrigger @unit(s) = default(0.16s);
        double railwayA3MinNeighborRsrp @unit(dB) = default(-130dB);

        //# Mobility and serving-cell statistics
        @signal[distance];
        @statistic[distance](title="distance between UE and serving base station"; unit="meters"; source="distance"; record=mean,vector);

        @signal[servingCell];
        @statistic[servingCell](title="ID of the serving cell for the UE"; unit=""; source="servingCell"; record=vector);

        //# Railway/RL handover observation statistics
        @signal[servingRsrp];
        @statistic[servingRsrp](title="RSRP from the serving cell"; unit="dB"; source="servingRsrp"; record=mean,vector);

        @signal[neighborRsrp];
        @statistic[neighborRsrp](title="RSRP from the strongest non-serving cell"; unit="dB"; source="neighborRsrp"; record=mean,vector);

        @signal[deltaRsrp];
        @statistic[deltaRsrp](title="neighborRsrp minus servingRsrp"; unit="dB"; source="deltaRsrp"; record=mean,vector);

        @signal[bestNeighborCell];
        @statistic[bestNeighborCell](title="ID of the strongest non-serving cell"; unit=""; source="bestNeighborCell"; record=vector);

        @signal[uePositionX];
        @statistic[uePositionX](title="UE x coordinate"; unit="m"; source="uePositionX"; record=mean,vector);

        @signal[ueSpeed];
        @statistic[ueSpeed](title="UE speed magnitude"; unit="mps"; source="ueSpeed"; record=mean,vector);

        @signal[timeSinceLastHandover];
        @statistic[timeSinceLastHandover](title="time elapsed since the last completed handover"; unit="s"; source="timeSinceLastHandover"; record=mean,vector);

        //# CQI statistics
        @signal[averageCqiDl];
        @statistic[averageCqiDl](title="Average Cqi reported in DL"; unit="cqi"; source="averageCqiDl"; record=mean,count,vector);
        @signal[averageCqiUl];
        @statistic[averageCqiUl](title="Average Cqi reported in UL"; unit="cqi"; source="averageCqiUl"; record=mean,count,vector);
}
