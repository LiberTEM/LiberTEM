// tslint:disable-next-line:ban-types
type ImmutablePrimitive = undefined | null | boolean | string | number | Function;

export type Immutable<T> =
    T extends ImmutablePrimitive ? T :
    T extends Array<infer U> ? ImmutableArray<U> :
    T extends Map<infer K, infer V> ? ImmutableMap<K, V> :
    T extends Set<infer M> ? ImmutableSet<M> : ImmutableObject<T>;

export type ImmutableArray<T> = ReadonlyArray<Immutable<T>>;
export type ImmutableMap<K, V> = ReadonlyMap<Immutable<K>, Immutable<V>>;
export type ImmutableSet<T> = ReadonlySet<Immutable<T>>;
export type ImmutableObject<T> = { readonly [K in keyof T]: Immutable<T[K]> };


type DeepReadonly<T> =
    T extends Array<infer R> ? DeepReadonlyArray<R> :
    // tslint:disable-next-line:ban-types
    T extends Function ? T :
    T extends object ? DeepReadonlyObject<T> :
    T;

interface DeepReadonlyArray<T> extends ReadonlyArray<DeepReadonly<T>> { }

type DeepReadonlyObject<T> = {
    readonly [P in keyof T]: DeepReadonly<T[P]>;
};

interface IdMap<R> {
    [s: string]: R
}

export interface ById<R> {
    ids: string[],
    byId: IdMap<R>,
};

export type ByIdReadOnly<R> = DeepReadonly<ById<R>>;

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
    const ids: string[] = state.ids.filter(id => pred(state.byId[id]));
    const byId: IdMap<R> = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return {
        byId,
        ids,
    };
}

export function filterWithPredReadOnly<R>(state: ByIdReadOnly<R>, pred: Predicate<DeepReadonly<R>>): ByIdReadOnly<R> {
    const ids: DeepReadonly<string[]> = state.ids.filter(id => pred(state.byId[id]));
    const byId: DeepReadonly<IdMap<R>> = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return {
        byId,
        ids,
    };
}