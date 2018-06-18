(function(Redux, ReduxSaga) {
    // TODO: replace with https://www.npmjs.com/package/uuid
    function uuid(a){
        // sorry... :)
        // stolen from https://gist.github.com/jed/982883
        return a?(a^Math.random()*16>>a/4).toString(16):([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,uuid)
    }

    function makeEnum(prefix, ...names) {
        return names.reduce((acc, curr) => Object.assign(acc, {[curr]: `${prefix}.${curr}`}), {});
    }

    const RESULT_STREAM = makeEnum("RESULT_STREAM",
        "CONNECT",
        "OPEN",
        "CLOSED",
        "ERROR",
        "MESSAGE",
        "HEAD_MESSAGE",
        "PART_MESSAGE",
        "AGGREGATE_MESSAGE",
    );

    function resultStreamMessage(msg, aggregate=null) {
        var parsed, raw;
        if(msg.data instanceof Blob) {
            // TODO: revokeObjectURL to free up unused images!
            // ref. https://developer.mozilla.org/en-US/docs/Web/API/URL/revokeObjectURL
            raw = URL.createObjectURL(msg.data);
            parsed = {};
        } else {
            raw = msg.data;
            parsed = JSON.parse(msg.data);
        }
        let payload = {msg: parsed, raw};

        if(aggregate === null) {
            return {
                type: RESULT_STREAM.MESSAGE,
                payload,
            };
        } else {
            let type = aggregate.head ? RESULT_STREAM.HEAD_MESSAGE : RESULT_STREAM.PART_MESSAGE;
            return {
                type,
                payload,
                meta: {
                    aggregate,
                }
            };
        }
    }

    function connectWS(/* TODO: addr, configuration */ dispatch) {
        function onMessage(msg) {
            if (msg.data instanceof Blob) {
                // assume this is a message part:
                dispatch(resultStreamMessage(msg, {
                    head: false,
                }));
            } else {
                var parsed = JSON.parse(msg.data);

                if(parsed.followup && parsed.followup.numMessages > 0) {
                    dispatch(resultStreamMessage(msg, {
                        head: true,
                        numParts: parsed.followup.numMessages,
                        type: RESULT_STREAM.AGGREGATE_MESSAGE,
                    }));
                } else {
                    dispatch(resultStreamMessage(msg));
                }
            }
        }

        // our protocol is started by the INITIAL_STATE message from the server, so we don't
        // strictly need this handler, but it can be useful as feedback for the user
        function onOpen() {
            dispatch({type: RESULT_STREAM.OPEN});
        }

        function onClose() {
            dispatch({type: RESULT_STREAM.CLOSED});
            // TODO: try to reconnect?
            // maybe handle this outside of connectWS, in a saga
        }

        function onError(err) {
            dispatch({type: RESULT_STREAM.ERROR, payload: {err}});
            // TODO: try to reconnect?
            // maybe handle this outside of connectWS, in a saga
        }

        let ws = new WebSocket("ws://localhost:9000/events/");
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
    }

    function openDataset({dataset}) {
        let datasetId = uuid();
        let payload = {
            dataset,
        };
        return fetch(`/datasets/${datasetId}/`, {
            method: "PUT",
            credentials: "same-origin",
            body: JSON.stringify(payload),
        }).then(r => r.json());
    }

    function startJob({masks, dataset}) {
        let jobId = uuid();
        let payload = {
            job: {
                dataset,
                masks,
            }
        }
        return fetch(`/jobs/${jobId}/`, {
            method: "PUT",
            credentials: "same-origin",
            body: JSON.stringify(payload),
        }).then(r => r.json());
    }

    async function initialize() {
        let dsResponseEMPAD1 = openDataset({
            dataset: {
                name: "test dataset",
                type: "HDFS",
                path: "/test/index.json",
                tileshape: [1, 8, 128, 128],
            },
        });
        let dsResponseEMPAD2 = openDataset({
            dataset: {
                name: "e field mapping acquisition 8",
                type: "HDFS",
                path: "/e-field-acquisition_8_0tilt_0V/index.json",
                tileshape: [1, 8, 128, 128],
            },
        });
        let dsResponseEMPAD3 = openDataset({
            dataset: {
                name: "e field mapping acquisition 10",
                type: "HDFS",
                path: "/e-field-acquisition_10_0tilt_40V/index.json",
                tileshape: [1, 8, 128, 128],
            },
        });
        let dss = {
            test: await dsResponseEMPAD1,
            eField1: await dsResponseEMPAD2,
            eField2: await dsResponseEMPAD3,
        };

        let job = await startJob({
            dataset: dss.test.dataset,
            masks: [
                {shape: "ring", cx: 64, cy: 65, ri: 0, ro: 5},
                {shape: "ring", cx: 64, cy: 65, ri: 0, ro: 45},
                {shape: "ring", cx: 64, cy: 65, ri: 50, ro: 63},
            ],
        });
    }

    function* initSaga(action) {
        console.log(`initSaga: action=`, action);
        if(action.payload.msg.messageType == "INITIAL_STATE") {
            yield ReduxSaga.effects.call(initialize);
        }
    }

    function* aggregateSaga(action) {
        console.log(`aggregateSaga: action=`, action);
        let headType = action.payload.head.payload.msg.messageType;
        if(headType === "TASK_RESULT" || headType == "FINISH_JOB") {
            let images = action.payload.parts.map((part) => {
                return part.payload.raw;
            });
            yield ReduxSaga.effects.call(showImages, action.payload, images);
        }
    }

    function showImages(payload, images) {
        console.log("showImages:", payload);
        images.forEach((imageData, idx) => {
            // TODO: blit to a canvas: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
            let img = document.getElementById(`image-${idx}`);
            img.src = imageData;
        });
    }

    function* webSocketSaga() {
        // TODO: replace dispatch function with something saga-aware?
        // how do we bridge the callback and generator worlds?
        // check redux-saga docs again

        while(true) {
            let cleanup = yield ReduxSaga.effects.call(connectWS, store.dispatch);
            let action = yield ReduxSaga.effects.take([
                RESULT_STREAM.OPEN,
                RESULT_STREAM.CLOSED,
            ]);
            if(action.type == RESULT_STREAM.OPEN) {
                yield ReduxSaga.effects.take([
                    RESULT_STREAM.CLOSED,
                    RESULT_STREAM.ERROR,
                ]);
            }
            cleanup();
            yield ReduxSaga.delay(1000);
        }
    }

    function* messageSaga() {
        console.log(`messageSaga initializing`);
        yield ReduxSaga.effects.takeEvery(RESULT_STREAM.MESSAGE, initSaga);
        yield ReduxSaga.effects.takeEvery(RESULT_STREAM.AGGREGATE_MESSAGE, aggregateSaga);
    }

    function *rootSaga() {
        yield ReduxSaga.effects.all([
            messageSaga(),
            webSocketSaga(),
        ]);
    }

    let rootReducer = (state = {}, action) => state;

    let sagaMiddleware = ReduxSaga.default();

    let { createStore, applyMiddleware, compose } = Redux;

    const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;

    var store = createStore(rootReducer, null, composeEnhancers(
        applyMiddleware(
            aggregateActions,
            sagaMiddleware,
        )
    ));

    sagaMiddleware.run(rootSaga);
})(Redux, ReduxSaga);
