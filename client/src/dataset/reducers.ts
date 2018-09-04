import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { constructById, filterWithPred, insertById, updateById } from "../helpers/reducerHelpers";
import { DatasetState, DatasetStatus } from "../messages";
import * as datasetActions from './actions';
import { DatasetsState, OpenDatasetState } from "./types";

const initialDatasetState: DatasetsState = {
    byId: {},
    ids: [],
};

export function datasetReducer(state = initialDatasetState, action: AllActions): DatasetsState {
    switch (action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            // FIXME: without type annotation, missing attributes in reducer state are not detected
            const datasets: DatasetState[] = action.payload.datasets.map(ds => ({
                id: ds.id,
                status: DatasetStatus.OPEN,
                params: ds.params,
                diagnostics: ds.diagnostics,
            }));
            return {
                byId: constructById(datasets, ds => ds.id),
                ids: datasets.map(ds => ds.id),
            }
        }
        case datasetActions.ActionTypes.CREATE: {
            const ds = { ...action.payload.dataset, status: DatasetStatus.OPENING };
            return insertById(state, action.payload.dataset.id, ds);
        }
        case datasetActions.ActionTypes.CREATED: {
            const ds = { ...action.payload.dataset, status: DatasetStatus.OPEN };
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
    formVisible: false,
    formPath: undefined,
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
    }
    return state;
}