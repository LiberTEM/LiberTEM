
export function parseNumList(nums: string) {
    return nums.split(",").map(part => +part);
}

export function getInitial<T extends object, K extends keyof T>(key: K, otherwise: string, values?: T): string {
    if (!values) {
        return otherwise;
    }
    return (values[key] ? values[key] : otherwise).toString();
}