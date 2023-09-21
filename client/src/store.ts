import { applyMiddleware, combineReducers, compose, createStore } from "redux";
import createSagaMiddleware from 'redux-saga';
import { analysisReducer } from "./analysis/reducers";
import { directoryBrowserReducer } from './browser/reducers';
import { channelStatusReducer } from "./channel/reducers";
import { clusterConnectionReducer } from "./cluster/reducers";
import { compoundAnalysisReducer } from "./compoundAnalysis/reducers";
import { configReducer } from "./config/reducers";
import { datasetReducer, openDatasetReducer } from "./dataset/reducers";
import { errorReducer } from "./errors/reducers";
import { jobReducer } from "./job/reducers";
import { progressReducer } from "./progress/reducers";

export const rootReducer = combineReducers({
    compoundAnalyses: compoundAnalysisReducer,
    analyses: analysisReducer,
    channelStatus: channelStatusReducer,
    clusterConnection: clusterConnectionReducer,
    datasets: datasetReducer,
    openDataset: openDatasetReducer,
    jobs: jobReducer,
    progress: progressReducer,
    errors: errorReducer,
    config: configReducer,
    browser: directoryBrowserReducer,
})

export type RootReducer = ReturnType<typeof rootReducer>;

export const sagaMiddleware = createSagaMiddleware();

declare global {
    interface Window { __REDUX_DEVTOOLS_EXTENSION_COMPOSE__: typeof compose }
}

// eslint-disable-next-line no-underscore-dangle
const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

export const store = createStore(rootReducer, composeEnhancers(
    applyMiddleware(
        sagaMiddleware,
    )
));
