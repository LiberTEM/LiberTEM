import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { Dataset } from './types';


export enum ActionTypes {
    CREATE = 'DATASET_CREATE',
    CREATED = 'DATASET_CREATED',
}

export const Actions = {
    // tslint:disable object-literal-sort-keys
    create: (dataset: Dataset) => createAction(ActionTypes.CREATE, dataset),
    created: (dataset: Dataset) => createAction(ActionTypes.CREATED, dataset),
    // tslint:enable
}

export type Actions = ActionsUnion<typeof Actions>;
