import * as analysisActions from './analysis/actions';
import * as browserActions from './browser/actions';
import * as channelActions from './channel/actions';
import * as clusterActions from './cluster/actions';
import * as compoundAnalysisActions from './compoundAnalysis/actions';
import * as configActions from './config/actions';
import * as datasetActions from './dataset/actions';
import * as errorActions from './errors/actions';
import * as jobActions from './job/actions';

export type AllActions = channelActions.Actions
    | datasetActions.Actions
    | analysisActions.Actions
    | clusterActions.Actions
    | configActions.Actions
    | errorActions.Actions
    | browserActions.Actions
    | jobActions.Actions
    | compoundAnalysisActions.Actions