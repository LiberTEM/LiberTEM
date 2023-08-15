import { test, expect, describe } from 'vitest';

import freeze from 'deep-freeze';
import { ById, insertById, updateById } from "../reducerHelpers";

interface O {
    payload: string,
}

describe('insertById', () => {
    test('works on empty initial state', () => {
        const state: ById<O> = {
            ids: [],
            byId: {},
        }

        freeze(state);

        const newState = insertById(state, "42", { payload: "wat" });

        expect(newState).toEqual({
            ids: ["42"],
            byId: { "42": { payload: "wat" } }
        })
    })
})

describe('updateById', () => {
    test('updates the payload', () => {
        const state = {
            ids: ["42"],
            byId: { "42": { payload: "wat" } }
        }

        freeze(state);

        const newState = updateById(state, "42", { payload: "hmm" });

        expect(newState).toEqual({
            ids: ["42"],
            byId: { "42": { payload: "hmm" } }
        })
    })
})