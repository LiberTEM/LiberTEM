import { Channel, delay, END, eventChannel } from "redux-saga";
import { call, fork, put, take } from "redux-saga/effects";
import * as channelActions from "./actions";
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
                case fromMessages.MessageTypes.OPEN: {
                    yield put(channelActions.Actions.open(timestamp));
                    break;
                }
                case fromMessages.MessageTypes.CLOSE: {
                    yield put(channelActions.Actions.close(timestamp));
                    break;
                }
                case fromMessages.MessageTypes.ERROR: {
                    yield put(channelActions.Actions.error(msg.msg, timestamp));
                    break;
                }
                case fromMessages.MessageTypes.INITIAL_STATE: {
                    yield put(channelActions.Actions.initialState(msg.jobs, msg.datasets, timestamp));
                    break;
                }
                case fromMessages.MessageTypes.START_JOB: {
                    yield put(channelActions.Actions.startJob(msg.job, timestamp))
                    break;
                }
                case fromMessages.MessageTypes.FINISH_JOB: {
                    yield call(handleFinishJob, msg, socketChannel, timestamp);
                    break;
                }
                case fromMessages.MessageTypes.TASK_RESULT: {
                    yield call(handleTaskResult, msg, socketChannel, timestamp)
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

export function* handleTaskResult(msg: ReturnType<typeof fromMessages.Messages.taskResult>, socketChannel: SocketChannel, timestamp: number) {
    const parts: fromMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map(part => ({ imageURL: part.objectURL }));
    yield put(channelActions.Actions.taskResult(msg.job, images, timestamp));
}

export function* handleFinishJob(msg: ReturnType<typeof fromMessages.Messages.finishJob>, socketChannel: SocketChannel, timestamp: number) {
    const parts: fromMessages.BinaryMessage[] = yield call(handleBinaryParts, msg.followup.numMessages, socketChannel);
    const images = parts.map(part => ({ imageURL: part.objectURL }));
    yield put(channelActions.Actions.finishJob(msg.job, images, timestamp));
}