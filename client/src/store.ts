import { combineReducers } from "redux";
import { channelStatusReducer } from "./channel/reducers";
import { datasetReducer } from "./dataset/reducers";
import { jobReducer } from "./job/reducers";

export const rootReducer = combineReducers({
    channelStatus: channelStatusReducer,
    dataset: datasetReducer,
    job: jobReducer,
})

export type RootReducer = ReturnType<typeof rootReducer>;