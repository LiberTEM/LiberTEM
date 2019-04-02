import { Channel, delay, END, eventChannel } from "redux-saga";
import { call, fork, put, take } from "redux-saga/effects";
import uuid from 'uuid/v4';
import * as datasetActions from '../dataset/actions';
import * as channelActions from "./actions";
import * as channelMessages from './messages';

type SocketChannel = Channel<channelMessages.Messages>;


/**
 * create typesafe messages from the websocket messages
 * 
 * also creates some synthetic events like open, close, error
 */
function createWebSocketChannel(/* addr */): SocketChannel {
    return eventChannel(emit => {
        function onMessage(msg: MessageEvent) {
            if (msg.data instanceof Blob) {
                // TODO: cleanup createObjectURL results somewhere
                emit(channelMessages.Messages.binary(URL.createObjectURL(msg.data)));
            } else {
                const parsed = JSON.parse(msg.data) as channelMessages.Messages;
                emit(parsed);
            }
        }

        function onOpen() {
            emit(channelMessages.Messages.open());
        }

        function onClose() {
            emit(channelMessages.Messages.close());
            emit(END);
        }

        function onError(err: Event) {
            emit(channelMessages.Messages.error("Error in weboscket connection"));
        }

        const ws = new WebSocket(`ws://${window.location.hostname}:${window.location.port}/api/events/`);
        ws.addEventListener("message", onMessage);
        ws.addEventListener("open", onOpen);
        ws.addEventListener("close", onClose);
        ws.addEventListener("error", onError);

        // return cleanup function:
        return () => {
            ws.removeEventListener("message", onMessage);
            ws.removeEventListener("open", onOpen);
            ws.removeEventListener("close", onClose);
            ws.removeEventListener("error", onError);
            // TODO: close connection if still open
            // (or is it guaranteed that if an error was thrown, the connection is closed?)
        };
    });
}

/**
 * handles the connection lifecycle for our websocket
 */
export function* webSocketSaga() {
    while (true) {
        const socketChannel = yield call(createWebSocketChannel);
        yield fork(actionsFromChannel, socketChannel);
        const action: channelActions.Actions = yield take([
            channelActions.ActionTypes.OPEN,
            channelActions.ActionTypes.CLOSE,
        ]);
        if (action.type === channelActions.ActionTypes.OPEN) {
            yield take([
                channelActions.ActionTypes.CLOSE,
                channelActions.ActionTypes.ERROR,
            ]);
        }
        yield delay(1000);
    }
}

/**
 * translates the messages from the channel to redux actions, handles aggregation etc.
 */
export function* actionsFromChannel(socketChannel: SocketChannel) {
    try {
        while (true) {
            const msg = yield take(socketChannel);
            const timestamp = Date.now();
            switch (msg.messageType) {
                case channelMessages.MessageTypes.OPEN: {
                    yield put(channelActions.Actions.open(timestamp));
                    break;
                }
                case channelMessages.MessageTypes.CLOSE: {
                    yield put(channelActions.Actions.close(timestamp));
                    break;
                }
                case channelMessages.MessageTypes.ERROR: {
                    const id = uuid();
                    yield put(channelActions.Actions.error(msg.msg, timestamp, id));
                    break;
                }
                case channelMessages.MessageTypes.INITIAL_STATE: {
                    yield put(channelActions.Actions.initialState(msg.jobs, msg.datasets, timestamp));
                    break;
                }
                case channelMessages.MessageTypes.JOB_STARTED: {
                    yield put(channelActions.Actions.jobStarted(msg.job, msg.details.dataset, timestamp));
                    break;
                }
                case channelMessages.MessageTypes.FINISH_JOB: {
                    yield call(handleFinishJob, msg, socketChannel, timestamp);
                    break;
                }
                case channelMessages.MessageTypes.TASK_RESULT: {
                    yield call(handleTaskResult, msg, socketChannel, timestamp);
                    break;
                }
                case channelMessages.MessageTypes.DELETE_DATASET: {
                    yield put(datasetActions.Actions.deleted(msg.dataset));
                    break;
                }
                case channelMessages.MessageTypes.JOB_ERROR: {
                    const id = uuid();
                    yield put(channelActions.Actions.jobError(msg.job, msg.msg, id, timestamp));
                }
            }
        }
    } finally {
        // disconnected
    }
}

export function* handleBinaryParts(numParts: number, socketChannel: SocketChannel) {
    const parts: channelMessages.BinaryMessage[] = [];
    while (parts.length < numParts) {
        const binMsg = yield take(socketChannel)
        parts.push(binMsg);
    }
    return parts;
}

export function* handleTaskResult(msg: ReturnType<typeof channelMessages.Messages.taskResult>, socketChannel: SocketChannel, timestamp: number) {
    const parts: channelMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map((part, idx) => ({ imageURL: part.objectURL, description: msg.followup.descriptions[idx] }));
    yield put(channelActions.Actions.taskResult(msg.job, images, timestamp));
}

export function* handleFinishJob(msg: ReturnType<typeof channelMessages.Messages.finishJob>, socketChannel: SocketChannel, timestamp: number) {
    const parts: channelMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map((part, idx) => ({ imageURL: part.objectURL, description: msg.followup.descriptions[idx] }));
    yield put(channelActions.Actions.finishJob(msg.job, images, timestamp));
}