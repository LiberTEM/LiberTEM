import { Channel, delay, END, eventChannel } from "redux-saga";
import { call, fork, put, take } from "redux-saga/effects";
import * as fromActions from "./actions";
import * as fromMessages from './messages';

type SocketChannel = Channel<fromMessages.Messages>;


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
                emit(fromMessages.Messages.binary(URL.createObjectURL(msg.data)));
            } else {
                const parsed = JSON.parse(msg.data) as fromMessages.Messages;
                emit(parsed);
            }
        }

        // our protocol is started by the INITIAL_STATE message from the server, so we don't
        // strictly need this handler, but it can be useful as feedback for the user
        function onOpen() {
            emit(fromMessages.Messages.open());
        }

        function onClose() {
            emit(fromMessages.Messages.close());
            emit(END);
        }

        function onError(err: Event) {
            emit(fromMessages.Messages.error(err.toString()));
        }

        // FIXME: hardcoded server location
        const ws = new WebSocket("ws://localhost:9000/events/");
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
        const action: fromActions.Actions = yield take([
            fromActions.ActionTypes.OPEN,
            fromActions.ActionTypes.CLOSE,
        ]);
        if (action.type === fromActions.ActionTypes.OPEN) {
            yield take([
                fromActions.ActionTypes.CLOSE,
                fromActions.ActionTypes.ERROR,
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
            switch (msg.messageType) {
                case fromMessages.MessageTypes.OPEN: {
                    yield put(fromActions.Actions.open());
                    break;
                }
                case fromMessages.MessageTypes.CLOSE: {
                    yield put(fromActions.Actions.close());
                    break;
                }
                case fromMessages.MessageTypes.ERROR: {
                    yield put(fromActions.Actions.error(msg.msg));
                    break;
                }
                case fromMessages.MessageTypes.INITIAL_STATE: {
                    yield put(fromActions.Actions.initialState(msg.jobs, msg.datasets));
                    break;
                }
                case fromMessages.MessageTypes.START_JOB: {
                    yield put(fromActions.Actions.startJob(msg.job))
                    break;
                }
                case fromMessages.MessageTypes.FINISH_JOB: {
                    yield call(handleFinishJob, msg, socketChannel);
                    break;
                }
                case fromMessages.MessageTypes.TASK_RESULT: {
                    yield call(handleTaskResult, msg, socketChannel)
                    break;
                }
            }
        }
    } finally {
        // disconnected
    }
}

export function* handleBinaryParts(numParts: number, socketChannel: SocketChannel) {
    const parts: fromMessages.BinaryMessage[] = [];
    while (parts.length < numParts) {
        const binMsg = yield take(socketChannel)
        parts.push(binMsg);
    }
    return parts;
}

export function* handleTaskResult(msg: ReturnType<typeof fromMessages.Messages.taskResult>, socketChannel: SocketChannel) {
    const parts : fromMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map(part => ({imageURL: part.objectURL}));
    yield put(fromActions.Actions.taskResult(msg.job, images));
}

export function* handleFinishJob(msg: ReturnType<typeof fromMessages.Messages.finishJob>, socketChannel: SocketChannel) {
    const parts : fromMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map(part => ({imageURL: part.objectURL}));
    yield put(fromActions.Actions.finishJob(msg.job, images));
}