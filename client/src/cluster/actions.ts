import { ActionsUnion, createAction } from '../helpers/actionHelpers';
import { ConnectRequestParams } from '../messages';

export enum ActionTypes {
    NOT_CONNECTED = 'CLUSTER_NOT_CONNECTED',
    CONNECT = 'CLUSTER_CONNECT',
    CONNECTING = 'CLUSTER_CONNECTING',
    CONNECTED = 'CLUSTER_CONNECTED',
    ERROR = 'CLUSTER_ERROR'
    
}

export const Actions = {
    notConnected: () => createAction(ActionTypes.NOT_CONNECTED),
    connect: (params: ConnectRequestParams) => createAction(ActionTypes.CONNECT, { params },),
    connecting: () => createAction(ActionTypes.CONNECTING),
    connected: (params: ConnectRequestParams ) => createAction(ActionTypes.CONNECTED, { params }),
    error: (msg: string, timestamp: number, id: string) => createAction(ActionTypes.ERROR, { msg, timestamp, id }),
}

export type Actions = ActionsUnion<typeof Actions>;
