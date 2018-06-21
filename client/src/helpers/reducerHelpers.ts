export interface ById<P> {
    ids: string[],
    byId: { [s: string]: P },
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

export function updateById<P, S extends ById<P>>(state: S, id: string, partialData: Partial<P>): S {
    const newObj = Object.assign({}, state.byId[id], partialData);
    const newById = Object.assign({}, state.byId, { [id]: newObj });
    return Object.assign({}, state, { byId: newById });
}

export function insertById<P extends object, S extends ById<P>>(state: S, id: string, data: P): S {
    const newById = Object.assign({}, state.byId, { [id]: data });
    const newIds = [...state.ids, id];
    return Object.assign({}, state, { byId: newById, ids: newIds });
}

export function getById<P extends object, S extends ById<P>>(items: P[], key: (k: P) => string) {
    return items.reduce((acc, item) => Object.assign(acc, {
        [key(item)]: item,
    }), {});
}