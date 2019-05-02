
export function parseNumList(nums: string) {
    return nums.split(",").map(part => +part);
}

export function getInitial<T extends object, K extends keyof T, V>(key: K, otherwise: V, values?: T): V | T[K] {
    if (!values) {
        return otherwise;
    }
    const res = values[key] !== undefined ? values[key] : otherwise;
    return res;
}