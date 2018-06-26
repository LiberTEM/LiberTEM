export function assertNotReached(message: string): never {
    throw new Error(message);
}