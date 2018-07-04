import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { ConnectRequestParams } from '../messages';

export enum ActionTypes {
    NOT_CONNECTED = 'CLUSTER_NOT_CONNECTED',
    CONNECT = 'CLUSTER_CONNECT',
    CONNECTED = 'CLUSTER_CONNECTED',
}

export const Actions = {
    notConnected: () => createAction(ActionTypes.NOT_CONNECTED),
    connect: (params: ConnectRequestParams) => createAction(ActionTypes.CONNECT, { params }),
    connected: (params: ConnectRequestParams) => createAction(ActionTypes.CONNECTED, { params }),
}

export type Actions = ActionsUnion<typeof Actions>;
