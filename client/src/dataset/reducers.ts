import { AllActions } from "../actions";
import * as channelActions from '../channel/actions';
import { ById, constructById, insertById } from "../helpers/reducerHelpers";
import { Dataset } from "../messages";
import * as datasetActions from './actions';

export type DatasetState = ById<Dataset>;

const initialDatasetState: DatasetState = {
    byId: {},
    ids: [],
};

export function datasetReducer(state = initialDatasetState, action: AllActions) {
    switch (action.type) {
        case channelActions.ActionTypes.INITIAL_STATE: {
            // FIXME: without type annotation, missing attributes in reducer state are not detected
            const datasets: Dataset[] = action.payload.datasets.map(ds => ({
                id: ds.id,
                name: ds.name,
                path: ds.path,
                tileshape: ds.tileshape,
                type: ds.type,
                shape: ds.shape,
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