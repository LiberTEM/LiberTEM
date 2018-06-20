import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, getById } from "../helpers/reducerHelpers";
import * as datasetActions from './actions';
import { Dataset } from "./types";

export type DatasetState = ById<Dataset>;

const initialDatasetState : DatasetState = {
    byId: {},
    ids: [],
};

export function datasetReducer(state = initialDatasetState, action: AllActions) {
    switch(action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            const datasets = action.payload.datasets.map(ds => ({
                id: ds.dataset,
                name: ds.name,
                path: ds.path,
                tileshape: ds.tileshape,
                type: ds.type,
            }))
            return {
                byId: getById(datasets, ds => ds.id),
                ids: datasets.map(ds => ds.id),
            }
        }
        case datasetActions.ActionTypes.CREATED: {
            return state; // TODO
        }
    }
    return state;
}