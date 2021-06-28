import { ActionCreatorsMapObject } from "redux";

export interface Action<T extends string> {
    type: T
}

export interface ActionWithPayload<T extends string, P> extends Action<T> {
    payload: P
}

export interface ActionWithMeta<T extends string, P, M> extends ActionWithPayload<T, P> {
    meta: M
}

export function createAction<T extends string>(type: T): Action<T>;
export function createAction<T extends string, P>(type: T, payload: P): ActionWithPayload<T, P>;
export function createAction<T extends string, P, M>(type: T, payload: P, meta: M): ActionWithMeta<T, P, M>;
// eslint-disable-next-line prefer-arrow/prefer-arrow-functions
export function createAction<T extends string, P, M>(type: T, payload?: P, meta?: M) {
    if (meta === undefined && payload === undefined) {
        return { type };
    } else if (meta === undefined) {
        return { type, payload };
    } else {
        return { type, payload, meta }
    }
}

export type ActionsUnion<A extends ActionCreatorsMapObject> = ReturnType<A[keyof A]>