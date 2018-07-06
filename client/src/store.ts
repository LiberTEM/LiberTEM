import { combineReducers } from "redux";
import { analysisReducer } from "./analysis/reducers";
import { channelStatusReducer } from "./channel/reducers";
import { clusterConnectionReducer } from "./cluster/reducers";
import { datasetReducer } from "./dataset/reducers";
import { errorReducer } from "./errors/reducers";
import { jobReducer } from "./job/reducers";

export const rootReducer = combineReducers({
    analyses: analysisReducer,
    channelStatus: channelStatusReducer,
    clusterConnection: clusterConnectionReducer,
    dataset: datasetReducer,
    job: jobReducer,
    errors: errorReducer,
})

export type RootReducer = ReturnType<typeof rootReducer>;