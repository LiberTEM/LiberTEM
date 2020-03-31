export function getApiBasePath() {
    const loc = window.location.pathname;

    if(loc.endsWith('/')) {
        return `${loc}api/`;
    } else {
        return `${loc}/api/`;
    }
}

export function getApiWSURL() {
    const basePath = getApiBasePath();
    return `ws://${window.location.hostname}:${window.location.port}${basePath}events/`;
}

export async function genericDelete(path: string) {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}${path}`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await r.json();
}

export async function genericPut(path: string, payload: object) {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}${path}`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await r.json();
}