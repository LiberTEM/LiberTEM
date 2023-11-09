// eslint-disable-next-line @typescript-eslint/ban-types
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
    // eslint-disable-next-line @typescript-eslint/ban-types
    T extends Function ? T :
    // eslint-disable-next-line @typescript-eslint/ban-types
    T extends object ? DeepReadonlyObject<T> :
    T;

type DeepReadonlyArray<T> = ReadonlyArray<DeepReadonly<T>>

type DeepReadonlyObject<T> = {
    readonly [P in keyof T]: DeepReadonly<T[P]>;
};

interface IdMap<R> {
    [s: string]: R
}

export interface ById<R> {
    ids: string[],
    byId: IdMap<R>,
}

export type ByIdReadOnly<R> = DeepReadonly<ById<R>>;

export const updateById = <R>(state: ById<R>, id: string, partialRecord: Partial<R>): ById<R> => {
    const newObj = Object.assign({}, state.byId[id], partialRecord);
    const newById = Object.assign({}, state.byId, { [id]: newObj });
    return Object.assign({}, state, { byId: newById });
}

export const insertById = <R>(state: ById<R>, id: string, record: R): ById<R> => {
    const newById = Object.assign({}, state.byId, { [id]: record });
    const newIds = [...state.ids, id];
    return { byId: newById, ids: newIds };
}

export const insertOrReplace = <R>(state: ById<R>, id: string, record: R): ById<R> => {
    const newById = Object.assign({}, state.byId, { [id]: record });
    // keep the order stable, and insert the new id only if needed
    let newIds;
    if(state.ids.includes(id)) {
        newIds = [...state.ids];
    } else {
        newIds = [...state.ids, id];
    }
    return Object.assign({}, state, { byId: newById, ids: newIds });
}

export const updateIfExists = <R>(state: ById<R>, id: string, record: R): ById<R> => {
    const newById = Object.assign({}, state.byId, { [id]: record });
    if(state.ids.includes(id)) {
        const newIds = [...state.ids, id];
        return Object.assign({}, state, { byId: newById, ids: newIds });
    } else {
        // don't do enything if the id is not found
        return state;
    }
}

export const removeById = <R>(state: ById<R>, id: string): ById<R> => {
    const {[id]: _, ...newById} = state.byId;
    const newIds = state.ids.filter(thisId => thisId !== id);
    return { byId: newById, ids: newIds };
}

export const constructById = <R>(items: R[], key: (k: R) => string): IdMap<R> => {
    const byId = items.reduce((acc, item) => Object.assign(acc, {
        [key(item)]: item,
    }), {} as IdMap<R>);
    return byId;
}

export type MapFn<R> = (item: R) => R;

export const updateWithMap = <R>(state: ById<R>, fn: MapFn<R>): ById<R> => {
    const byId: IdMap<R> = state.ids.reduce((acc, id) => Object.assign(acc, {
        [id]: fn(state.byId[id]),
    }), {});
    return {
        byId,
        ids: state.ids,
    };
}

export type Predicate<R> = (item: R) => boolean;

export const filterWithPred = <R>(state: ById<R>, pred: Predicate<R>): ById<R> => {
    const ids: string[] = state.ids.filter(id => pred(state.byId[id]));
    const byId: IdMap<R> = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return {
        byId,
        ids,
    };
}

export const filterWithPredReadOnly = <R>(state: ByIdReadOnly<R>, pred: Predicate<DeepReadonly<R>>): ByIdReadOnly<R> => {
    const ids: DeepReadonly<string[]> = state.ids.filter(id => pred(state.byId[id]));
    const byId: DeepReadonly<IdMap<R>> = ids.reduce((acc, id) => Object.assign(acc, {
        [id]: state.byId[id],
    }), {});
    return {
        byId,
        ids,
    };
}

export const toggleItemInList = <T>(list: T[], item: T): T[] => {
    if (list.includes(item)) {
        return list.filter(i => i !== item)
    } else {
        return [item, ...list];
    }
}