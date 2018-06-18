/*
 * Generic action aggregation middleware
 *
 * The idea is that there is a stream of actions, and sometimes a bunch of them
 * need to be aggregated to one "logical" action. This is accomplished by having
 * actions of the format:
 *
 * {
 *  type: SOME_ACTION_TYPE,
 *  meta: {
 *   aggregate: {
 *    head: true,
 *    numParts: N,
 *    type: AGG_ACTION_TYPE,
 *   },
 *  },
 *  payload: {...},
 * }
 *
 * Followed by actions like this:
 *
 * {
 *  type: OTHER_ACTION_TYPE,
 *  meta: {
 *   aggregate: {
 *    head: false,
 *   },
 *  }
 *  payload: {...},
 * }
 *
 * Which are then aggregated to actions like this:
 *
 * {
 *  type: AGG_ACTION_TYPE,
 *  payload: {
 *   head: {type: SOME_ACTION_TYPE, payload: {...}},
 *   parts: [{type: OTHER_ACTION_TYPE, payload: {...}}, ...],
 *  }
 * }
 *
 * Note that the number of parts needs to be known when the head
 * action is fired. Also, only one aggregation can be handeled at once,
 * otherwise we wouldn't know to which "stream" of actions a certain action
 * belongs. We could introduce a "origin" or streamId/streamKey/... for that later.
 * In the beginning, once stream is enough.
 */


function aggregateActions(store) {
    let ctx = {
        expectedParts: 0,
        parts: [],
        head: undefined,
    };

    return next => action => {
        if(!action.meta || !action.meta.aggregate) {
            return next(action);
        }
        var agg = action.meta.aggregate;
        if(ctx.expectedParts > 0) {
            // part
            if(agg.head !== false) {
                throw new Error(`unexpected state, was expecting a part (head=false), got head=${agg.head}`);
            }
            ctx.expectedParts -= 1;
            ctx.parts.push(action);
            if(ctx.expectedParts === 0) {
                let result = next({
                    type: ctx.head.meta.aggregate.type,
                    payload: {
                        head: ctx.head,
                        parts: ctx.parts,
                    },
                });
                ctx.parts = [];
                ctx.head = undefined;
                return result;
            }
        } else {
            // head
            if(agg.head !== true) {
                throw new Error(`unexpected state, was expecting a head (head=true), got head=${agg.head}`);
            }
            ctx.head = action;
            ctx.expectedParts = agg.numParts;
        }
        // swallow the action:
        return store.getState();
    };
}
