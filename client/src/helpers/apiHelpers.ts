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