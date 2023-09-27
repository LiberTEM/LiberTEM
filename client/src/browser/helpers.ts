// String.prototype.split, but with a limit that works like Python,
// meaning the rest is not cut off but appended to the last item.
// limit=1 means "one split", so an array with two values is returned,
// limit=-1 is not supported
export const splitLikePython = (input: string, sep: string, limit: number) => {
    if (!input.includes(sep)) {
        return [input];
    }

    const splitNoLimit = input.split(sep);
    const leftPart = splitNoLimit.slice(0, limit);
    const rightPart = splitNoLimit.slice(limit).join(sep);

    return [
        ...leftPart,
        rightPart,
    ]
}

interface URLActionOpen {
    action: 'open',
    path: string,
}

interface URLActionNone {
    action: 'none',
}

type URLAction = URLActionOpen | URLActionNone;

interface HashParameters {
    [key: string]: string,
}

export const parseHashParameters = (hash: string): HashParameters => {
    if (hash === "" || !hash.includes('=')) {
        return {};
    }

    const byParam = hash.split('&');

    return Object.fromEntries(
        byParam.map(paramStr => splitLikePython(paramStr, '=', 1))
    ) as HashParameters; // yuck, fromEntries doesn't propagate the types from its argument
}

export const getUrlAction = (): URLAction => {
    const hash = window.location.hash.slice(1);
    const params = parseHashParameters(hash);
    const action = params.action;

    if (Object.keys(params).length === 0 || action === undefined) {
        return { action: 'none' };
    }

    if (action === 'open' && params.path !== undefined && params.path !== "") {
        return { action: 'open', path: params.path };
    }

    return { 'action': 'none' };
}