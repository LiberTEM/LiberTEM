// TODO: replace with https://www.npmjs.com/package/uuid
function uuid(a){
    // sorry... :)
    // stolen from https://gist.github.com/jed/982883
    return a?(a^Math.random()*16>>a/4).toString(16):([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g,uuid)
}

function makeEnum(...names) {
    return names.reduce((acc, curr) => Object.assign(acc, {[curr]: curr}), {});
}

const CONN_STATES = makeEnum("INITIAL", "IDLE", "JOB_RUNNING", "WAIT_FOR_FOLLOWUP");
const DATA_TYPE = makeEnum("BINARY", "JSON");


class Observable {
    constructor() {
        this.callbacks = [];
    }

    subscribe(callback, predicate = (payload) => true) {
        this.callbacks.push({fn: callback, predicate: predicate});
    }

    unsubscribe(callback) {
        this.callbacks = this.callbacks.filter(cb => cb.fn !== callback);
    }

    call(payload) {
        this.callbacks.forEach(cb => {
            if(cb.predicate(payload)) {
                cb.fn(payload);
            }
        });
    }
}

class AggregateMessage {
    constructor(headMessage, followupMessages) {
        this.head = headMessage;
        this.followup = followupMessages;
    }
}

function Client(initCb, resultCb) {
    let events = {
        messages: new Observable(),
        connection: new Observable(),
    };

    let ctx = {
        followupMessages: [],
        expectedFollowup: 0,
    };

    let state = CONN_STATES.INITIAL;

    function assertState(expectedState) {
        if(state !== expectedState) {
            throw new Error(`invalid state, expected ${expectedState}, got ${state}`);
        }
    }

    function assert(bool) {
        if(!bool) {
            throw new Error(`assertion failed`);
        }
    }

    function onMessage(msg) {
        let dataType = msg.data instanceof Blob ? DATA_TYPE.BINARY : DATA_TYPE.JSON;

        if(ctx.expectedFollowup > 0) {
            assert(dataType == DATA_TYPE.BINARY);
            assertState(CONN_STATES.WAIT_FOR_FOLLOWUP);
            ctx.expectedFollowup -= 1;
            ctx.followupMessages.push(msg);
            if(ctx.expectedFollowup === 0) {
                events.messages.call(new AggregateMessage(ctx.headMessage, ctx.followupMessages));
                ctx.headMessage = undefined;
                state = ctx.nextState;
                resultCb(ctx.followupMessages);
                ctx.followupMessages = [];
            }
            console.log(`done processing binary msg, expectedFollowup=${ctx.expectedFollowup}`);
        } else {
            try {
                var parsed = JSON.parse(msg.data);
            } catch (e) {
                throw e;
            }

            if(parsed.followup && parsed.followup.numMessages > 0) {
                ctx.headMessage = msg;
                ctx.expectedFollowup = parsed.followup.numMessages;
            }

            let msgType = parsed.messageType;
            console.log(`received message of type ${msgType} in state ${state}`);
            switch(msgType) {
                case "INITIAL_STATE":
                    assertState(CONN_STATES.INITIAL);
                    initCb(msg);
                    state = CONN_STATES.IDLE;
                    break;
                case "START_JOB":
                    assertState(CONN_STATES.IDLE);
                    state = CONN_STATES.JOB_RUNNING;
                    break;
                case "TASK_RESULT":
                    assertState(CONN_STATES.JOB_RUNNING);
                    ctx.nextState = CONN_STATES.JOB_RUNNING;
                    state = CONN_STATES.WAIT_FOR_FOLLOWUP;
                    break;
                case "FINISH_JOB":
                    assertState(CONN_STATES.JOB_RUNNING);
                    ctx.nextState = CONN_STATES.IDLE;
                    state = CONN_STATES.WAIT_FOR_FOLLOWUP;
                    console.timeEnd("doStuff");
                    break;
                default:
                    throw new Error(`invalid message type: ${msgType}`);
            }
        }
    }

    let ws = new WebSocket("ws://localhost:9000/events/");
    ws.addEventListener("message", onMessage);

    return {
        events,
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

async function doStuff() {
    console.time("doStuff");
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

Client(
    () => doStuff(),
    (messages) => {
        console.log(messages);
        messages.forEach((message, idx) => {
            // TODO: blit to a canvas: https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/drawImage
            let objectURL = URL.createObjectURL(message.data);
            let img = document.getElementById(`image-${idx}`);
            img.src = objectURL;
            // TODO: revokeObjectURL to free up unused images!
            // ref. https://developer.mozilla.org/en-US/docs/Web/API/URL/revokeObjectURL
        });
    }
);
