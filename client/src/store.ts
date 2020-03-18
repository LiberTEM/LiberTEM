import { combineReducers } from "redux";
import { analysisReducer } from "./analysis/reducers";
import { directoryBrowserReducer } from './browser/reducers';
import { channelStatusReducer } from "./channel/reducers";
import { clusterConnectionReducer } from "./cluster/reducers";
import { compoundAnalysisReducer } from "./compoundAnalysis/reducers";
import { configReducer } from "./config/reducers";
import { datasetReducer, openDatasetReducer } from "./dataset/reducers";
import { errorReducer } from "./errors/reducers";
import { jobReducer } from "./job/reducers";

export const rootReducer = combineReducers({
    compoundAnalyses: compoundAnalysisReducer,
    analyses: analysisReducer,
    channelStatus: channelStatusReducer,
    clusterConnection: clusterConnectionReducer,
    datasets: datasetReducer,
    openDataset: openDatasetReducer,
    jobs: jobReducer,
    errors: errorReducer,
    config: configReducer,
    browser: directoryBrowserReducer,
})

export type RootReducer = ReturnType<typeof rootReducer>;