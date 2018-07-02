import * as _ from "lodash";

export function assertNotReached(message: string): never {
    throw new Error(message);
}

export function defaultDebounce<T extends (...args: any[]) => any>(fn: T, delay: number = 50) {
    return _.debounce(fn, delay, { maxWait: delay });
}

export function getEnumValues(e: any) {
    return Object.keys(e).filter(k => typeof e[k as any] === "string");
}