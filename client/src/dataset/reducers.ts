import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { constructById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import { Dataset, DatasetState, DatasetStatus } from "../messages";
import * as datasetActions from './actions';
import { DatasetsState, OpenDatasetState } from "./types";

const initialDatasetState: DatasetsState = {
    byId: {},
    ids: [],
};

export function datasetReducer(state = initialDatasetState, action: AllActions): DatasetsState {
    switch (action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            const datasets = action.payload.datasets.map(ds => Object.assign({}, ds, { status: DatasetStatus.OPEN }));
            return {
                byId: constructById(datasets, ds => ds.id),
                ids: datasets.map(ds => ds.id),
            }
        }
        case datasetActions.ActionTypes.CREATE: {
            const ds: Dataset = {
                ...action.payload.dataset,
                status: DatasetStatus.OPENING
            };
            return insertById(state, action.payload.dataset.id, ds);
        }
        case datasetActions.ActionTypes.CREATED: {
            const ds = Object.assign({}, action.payload.dataset, { status: DatasetStatus.OPEN });
            return updateById(state, action.payload.dataset.id, ds);
        }
        case datasetActions.ActionTypes.ERROR: {
            return filterWithPred(state, (r: DatasetState) => r.id !== action.payload.dataset);
        }
        case datasetActions.ActionTypes.DELETE: {
            return updateById(state, action.payload.dataset, { status: DatasetStatus.DELETING });
        }
        case datasetActions.ActionTypes.DELETED: {
            return filterWithPred(state, (r: DatasetState) => r.id !== action.payload.dataset);
        }
    }
    return state;
}

const initialOpenDatasetState: OpenDatasetState = {
    busy: false,
    busyPath: "",
    formVisible: false,
    formPath: "/",
    formInitialParams: undefined,
}

export function openDatasetReducer(state = initialOpenDatasetState, action: AllActions): OpenDatasetState {
    switch (action.type) {
        case datasetActions.ActionTypes.OPEN: {
            return {
                ...state,
                formVisible: true,
                formPath: action.payload.path,
                formInitialParams: action.payload.initialParams,
            };
        }
        case datasetActions.ActionTypes.CANCEL_OPEN: {
            return {
                ...state,
                formVisible: false,
            }
        }
        case datasetActions.ActionTypes.CREATE: {
            return {
                ...state,
                formVisible: false,
            }
        }
        case datasetActions.ActionTypes.DETECT: {
            return {
                ...state,
                busyPath: action.payload.path,
                busy: true,
            }
        }
        case datasetActions.ActionTypes.DETECTED:
        case datasetActions.ActionTypes.DETECT_FAILED: {
            return {
                ...state,
                busyPath: "",
                busy: false,
            }
        }
    }
    return state;
}