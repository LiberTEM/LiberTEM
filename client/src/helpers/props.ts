export type DispatchProps<T> = {
    [P in keyof T]: T[P]
}