export interface ById<R> {
    ids: string[],
    byId: { [s: string]: R },
};

/*
TODO: make ById DeepReadonly
import { DeepReadonly } from 'utility-types'
type Foo<P> = DeepReadonly<{
    byId: { [s: string]: P },
}>;

export function readOnlyWithExplicitType(foo: Foo<string>, key: string) {
    return foo.byId[key];
}

export function readOnlyWithGeneric<P>(foo: Foo<P>, key: string) {
    // error: Element implicitly has an 'any' type because type 'DeepReadonlyObject<{ [s: string]: P; }>' has no index signature.
    return foo.byId[key];
}
*/

export function updateById<R, S extends ById<R>>(state: S, id: string, partialRecord: Partial<R>): S {
    const newObj = Object.assign({}, state.byId[id], partialRecord);
    const newById = Object.assign({}, state.byId, { [id]: newObj });
    return Object.assign({}, state, { byId: newById });
}

export function insertById<R extends object, S extends ById<R>>(state: S, id: string, record: R): S {
    const newById = Object.assign({}, state.byId, { [id]: record });
    const newIds = [...state.ids, id];
    return Object.assign({}, state, { byId: newById, ids: newIds });
}

export function constructById<R extends object, S extends ById<R>>(items: R[], key: (k: R) => string) {
    return items.reduce((acc, item) => Object.assign(acc, {
        [key(item)]: item,
    }), {});
}

export type Predicate<R> = (item: R) => boolean;

export function filterWithPred<R, S extends ById<R>>(state: S, pred: Predicate<R>): S {
    const ids = state.ids.filter(id => pred(state.byId[id]));
    const byId = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return Object.assign({}, state, {
        byId,
        ids,
    });
}