import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { constructById, insertById } from "../helpers/reducerHelpers";
import { DatasetState } from "../messages";
import * as datasetActions from './actions';
import { DatasetsState } from "./types";

const initialDatasetState: DatasetsState = {
    byId: {},
    ids: [],
};

export function datasetReducer(state = initialDatasetState, action: AllActions) {
    switch (action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            // FIXME: without type annotation, missing attributes in reducer state are not detected
            const datasets: DatasetState[] = action.payload.datasets.map(ds => ({
                id: ds.id,
                name: ds.name,
                params: ds.params,
            }));
            return {
                byId: constructById(datasets, ds => ds.id),
                ids: datasets.map(ds => ds.id),
            }
        }
        case datasetActions.ActionTypes.CREATED: {
            return insertById(state, action.payload.dataset.id, action.payload.dataset);
        }
    }
    return state;
}