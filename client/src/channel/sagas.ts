import { END, eventChannel, EventChannel } from "redux-saga";
import { call, delay, fork, put, take } from "redux-saga/effects";
import { v4 as uuid } from 'uuid';
import * as datasetActions from "../dataset/actions";
import { getApiWSURL } from "../helpers/apiHelpers";
import * as channelActions from "./actions";
import * as channelMessages from "./messages";

type SocketChannel = EventChannel<channelMessages.Messages>;

/**
 * create typesafe messages from the websocket messages
 *
 * also creates some synthetic events like open, close, error
 */
const createWebSocketChannel = (/* addr */): SocketChannel => eventChannel((emit) => {
    const onMessage = (msg: MessageEvent) => {
        if (msg.data instanceof Blob) {
            // TODO: cleanup createObjectURL results somewhere
            emit(channelMessages.Messages.binary(URL.createObjectURL(msg.data)));
        } else {
            const parsed = JSON.parse(msg.data as string) as channelMessages.Messages;
            emit(parsed);
        }
    }

    const onOpen = () => {
        emit(channelMessages.Messages.open());
    }

    const onClose = () => {
        emit(channelMessages.Messages.close());
        emit(END);
    }

    const onError = () => {
        emit(channelMessages.Messages.error("Error in weboscket connection"));
    }

    const ws = new WebSocket(getApiWSURL());
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
})

/**
 * handles the connection lifecycle for our websocket
 */
export function* webSocketSaga() {
    while (true) {
        const socketChannel = (yield call(createWebSocketChannel)) as SocketChannel;
        yield fork(actionsFromChannel, socketChannel);
        const action = (yield take([channelActions.ActionTypes.OPEN, channelActions.ActionTypes.CLOSE])) as channelActions.Actions;
        if (action.type === channelActions.ActionTypes.OPEN) {
            const isShutdown = (yield take([
                channelActions.ActionTypes.CLOSE,
                channelActions.ActionTypes.ERROR,
                channelActions.ActionTypes.CLOSE_LOOP,
            ])) as channelActions.Actions;
            if (isShutdown.type === channelActions.ActionTypes.CLOSE_LOOP) {
                break;
            }
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
            const msg = (yield take(socketChannel)) as channelMessages.Messages;
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
                    yield put(channelActions.Actions.initialState(msg.jobs, msg.datasets, msg.compoundAnalyses, msg.analyses, timestamp));
                    break;
                }
                case channelMessages.MessageTypes.JOB_STARTED: {
                    yield put(channelActions.Actions.jobStarted(msg.job, timestamp));
                    break;
                }
                case channelMessages.MessageTypes.JOB_PROGRESS: {
                    yield put(channelActions.Actions.jobProgress(msg.job, msg.details));
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
                case channelMessages.MessageTypes.CREATE_DATASET: {
                    yield put(datasetActions.Actions.created(msg.details));
                    break;
                }
                case channelMessages.MessageTypes.DELETE_DATASET: {
                    yield put(datasetActions.Actions.deleted(msg.dataset));
                    break;
                }
                case channelMessages.MessageTypes.JOB_ERROR: {
                    const id = uuid();
                    yield put(channelActions.Actions.jobError(msg.job, msg.msg, id, timestamp));
                    break;
                }
                case channelMessages.MessageTypes.CANCEL_JOB_DONE: {
                    yield put(channelActions.Actions.cancelled(msg.job));
                    break;
                }
                case channelMessages.MessageTypes.CANCEL_JOB_FAILED: {
                    yield put(channelActions.Actions.cancelFailed(msg.job));
                    break;
                }
                case channelMessages.MessageTypes.SNOOZE: {
                    yield put(channelActions.Actions.snooze(timestamp));
                    break;
                }
                case channelMessages.MessageTypes.UNSNOOZE: {
                    yield put(channelActions.Actions.unsnooze(timestamp));
                    break;
                }
                case channelMessages.MessageTypes.UNSNOOZE_DONE: {
                    yield put(channelActions.Actions.unsnooze_done(timestamp));
                }                
                /*
                // FIXME: server needs to know about compount analyses
                case channelMessages.MessageTypes.ANALYSIS_CREATED: {
                    yield put(channelActions.Actions.analysisCreated(
                        msg.analysis,
                        msg.dataset,
                        msg.details,
                    ));
                    break;
                }
                case channelMessages.MessageTypes.ANALYSIS_UPDATED: {
                    yield put(channelActions.Actions.analysisUpdated(
                        msg.analysis,
                        msg.dataset,
                        msg.details,
                    ));
                    break;
                }
                case channelMessages.MessageTypes.ANALYSIS_REMOVED: {
                    yield put(channelActions.Actions.analysisRemoved(
                        msg.analysis,
                    ));
                    break;
                }
                */
            }
        }
    } finally {
        // disconnected
    }
}

export function* handleBinaryParts(numParts: number, socketChannel: SocketChannel) {
    const parts: channelMessages.BinaryMessage[] = [];
    while (parts.length < numParts) {
        const binMsg = (yield take(socketChannel)) as channelMessages.BinaryMessage;
        parts.push(binMsg);
    }
    return parts;
}

export function* handleTaskResult(msg: ReturnType<typeof channelMessages.Messages.taskResult>, socketChannel: SocketChannel, timestamp: number) {
    const parts = (yield call(handleBinaryParts, msg.followup.numMessages, socketChannel)) as channelMessages.BinaryMessage[];
    const images = parts.map((part, idx) => ({ imageURL: part.objectURL, description: msg.followup.descriptions[idx] }));
    yield put(channelActions.Actions.taskResult(msg.job, images, timestamp));
}

export function* handleFinishJob(msg: ReturnType<typeof channelMessages.Messages.finishJob>, socketChannel: SocketChannel, timestamp: number) {
    const parts = (yield call(handleBinaryParts, msg.followup.numMessages, socketChannel)) as channelMessages.BinaryMessage[];
    const images = parts.map((part, idx) => ({ imageURL: part.objectURL, description: msg.followup.descriptions[idx] }));
    yield put(channelActions.Actions.finishJob(msg.job, images, timestamp));
}
