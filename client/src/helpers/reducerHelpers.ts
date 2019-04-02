interface IdMap<R> {
    [s: string]: R
}

export interface ById<R> {
    ids: string[],
    byId: IdMap<R>,
};

// TODO: make ById DeepReadonly
// import { DeepReadonly } from 'utility-types'

export function updateById<R>(state: ById<R>, id: string, partialRecord: Partial<R>): ById<R> {
    const newObj = Object.assign({}, state.byId[id], partialRecord);
    const newById = Object.assign({}, state.byId, { [id]: newObj });
    return Object.assign({}, state, { byId: newById });
}

export function insertById<R>(state: ById<R>, id: string, record: R): ById<R> {
    const newById = Object.assign({}, state.byId, { [id]: record });
    const newIds = [...state.ids, id];
    return { byId: newById, ids: newIds };
}

export function constructById<R>(items: R[], key: (k: R) => string): IdMap<R> {
    const byId = items.reduce((acc, item) => Object.assign(acc, {
        [key(item)]: item,
    }), {} as IdMap<R>);
    return byId;
}

export type Predicate<R> = (item: R) => boolean;

export function filterWithPred<R>(state: ById<R>, pred: Predicate<R>): ById<R> {
    const ids = state.ids.filter(id => pred(state.byId[id]));
    const byId: IdMap<R> = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return {
        byId,
        ids,
    };
}