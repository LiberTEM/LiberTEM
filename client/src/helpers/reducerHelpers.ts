import { DeepReadonly } from 'utility-types'

export type ById<Payload> = DeepReadonly<{
    ids: string[],
    byId: { [s: string]: Payload },
}>;

export function updateById<Payload, State extends ById<Payload>>(state: State, id: string, partialData: Partial<Payload>): State {
    const newObj = Object.assign({}, state.byId[id], partialData);
    const newById = Object.assign({}, state.byId, {[id]: newObj});
    return Object.assign({}, state, {byId: newById});
}

export function insertById<Payload extends object, State extends ById<Payload>>(state: State, id: string, data: Payload): State {

    const newById = Object.assign({}, state.byId, {[id]: data});
    const newIds = [...state.ids, id];
    return Object.assign({}, state, {byId: newById, ids: newIds});
}

export function getById<Payload extends object, State extends ById<Payload>>(items: Payload[], key: (k: Payload) => string) {
    return items.reduce((acc, item) => Object.assign(acc, {
        [key(item)]: item,
    }), {});
}