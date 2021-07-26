export const getApiBasePath = () : string => {
    const loc = window.location.pathname;

    if(loc.endsWith('/')) {
        return `${loc}api/`;
    } else {
        return `${loc}/api/`;
    }
}

export const getApiWSURL = () : string => {
    const basePath = getApiBasePath();
    const isSecure = window.location.protocol === 'https:';
    const proto = isSecure ? 'wss' : 'ws';
    return `${proto}://${window.location.hostname}:${window.location.port}${basePath}events/`;
}

export const genericDelete = async <T>(path: string): Promise<T> => {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}${path}`, {
        credentials: "same-origin",
        method: "DELETE",
    });
    return await (r.json() as Promise<T>);
}


export const genericPut = async <Resp, Payload>(path: string, payload: Payload): Promise<Resp> => {
    const basePath = getApiBasePath();
    const r = await fetch(`${basePath}${path}`, {
        body: JSON.stringify(payload),
        credentials: "same-origin",
        method: "PUT",
    });
    return await (r.json() as Promise<Resp>);
}
