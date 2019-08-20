import * as _ from 'lodash';
import { MsgPartConfig } from "../messages";
import { ConfigParams, ConfigState, LocalConfig } from "./reducers";

const CONFIG_KEY = "LiberTEM.config";

export function joinPaths(config: ConfigState, ...parts: string[]) {
    const removeSep = (part: string) => part.replace(new RegExp(`${_.escapeRegExp(config.separator)}$`), "")
    parts = [removeSep(parts[0]), ...parts.slice(1)];
    return parts.map(part => part.trim()).join(config.separator);
}

export function mergeLocalStorage(serverConfig: MsgPartConfig): ConfigParams {
    const localSettings = window.localStorage.getItem(CONFIG_KEY);
    if (localSettings === null) {
        return Object.assign({}, serverConfig, getDefaultLocalConfig(serverConfig));
    }
    const localSettingsParsed = JSON.parse(localSettings);
    const defaultConfig = getDefaultLocalConfig(serverConfig);
    const mergedConfig = Object.assign({}, defaultConfig, serverConfig, localSettingsParsed);
    return mergedConfig;
}

type ConfigKeys = Array<Partial<keyof ConfigState>>;

export function setLocalStorage(config: ConfigState): void {
    
    const keys: ConfigKeys = ["cwd", "lastOpened", "fileHistory"];
    const localSettings = keys.reduce((acc, item) => {
        const configItem = config[item];
        acc[item] = configItem;
        return acc;
    }, {} as ConfigState);
    window.localStorage.setItem(CONFIG_KEY, JSON.stringify(localSettings));
}

export function clearLocalStorage(): void {
    window.localStorage.removeItem(CONFIG_KEY);
}

export function getDefaultLocalConfig(config: MsgPartConfig): LocalConfig {
    return {
        lastOpened: {},
        fileHistory: [],
    };
}

export function makeUnique<T>(inp: T[]): T[] {
    return inp.reduce((acc: T[], curr: T) => {
        if (acc.indexOf(curr) === -1) {
            return [...acc, curr];
        } else {
            return acc;
        }
    }, [] as T[]);
}