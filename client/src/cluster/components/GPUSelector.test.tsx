/* eslint-disable @typescript-eslint/no-empty-function */
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from 'vitest';
import { ConfigState } from '../../config/reducers';
import { ClusterTypes } from '../../messages';
import { GPUSelector } from './GPUSelector';


const defaultConfig: ConfigState = {
    version: "v0.1.2",
    revision: "abcdef",
    localCores: 7,
    separator: '/',
    cwd: '/data/',
    datasetTypes: {
        'MIB': {
            schema: {},
            default_io_backend: null,
            supported_io_backends: [],
        },
    },
    devices: {
        cpus: [0, 1, 2, 3, 4, 5, 6],
        cudas: [],
        has_cupy: false,
    },
    resultFileFormats: {
        NPZ: {
            identifier: 'NPZ',
            description: 'numpy format (.npz)',
        },
    },
    fileHistory: [],
    lastOpened: {},
    lastConnection: {
        type: ClusterTypes.LOCAL,
        address: 'tcp://localhost:8786',
        cudas: {},
    },
    starred: [],
    haveConfig: true,
};

const deepcopy = <T extends object>(o: T): T => JSON.parse(JSON.stringify(o)) as T;

describe('GPUSelector', () => {
    it('renders for zero cuda devices', () => {
        const config = deepcopy(defaultConfig);
        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [],
            has_cupy: false,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {},
        };
        const value = {
        };
        const setFieldValue = (_name: string, _value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );
        expect(screen.queryByText(/No cupy installation found/)).toBeNull();
        expect(screen.queryByText(/Number of workers for CUDA device/)).toBeNull();
        expect(screen.queryByText(/No CUDA devices found/)).toBeInTheDocument();
        expect(screen.queryByRole('spinbutton')).toBeNull();
    });

    it('renders warning if no cupy is found', () => {
        const config = deepcopy(defaultConfig);
        config.devices =  {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0, 1, 2],   // we have cuda devices, ...
            has_cupy: false,    // but no working cupy installation
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };
        const value = {
            0: 42,
            1: 1,
            2: 1,
        };
        const setFieldValue = (_name: string, _value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/No cupy installation found/)).toBeInTheDocument();
        expect(screen.getAllByText(/Number of workers for CUDA device/).length).toBe(3);
        screen.getAllByRole('spinbutton').forEach(btn => {
            expect(btn).toBeDisabled();
        });
    });

    it('renders for one cuda device', () => {
        const config = deepcopy(defaultConfig);
        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0],
            has_cupy: true,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };
        const value = {
            0: 42,  // initial value from lastConnection
        };
        const setFieldValue = (_name: string, _value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();
        expect(screen.getByRole('spinbutton')).toBeInTheDocument();
    });

    it('renders for more cuda devices', () => {
        const config = deepcopy(defaultConfig);
        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0, 1],
            has_cupy: true,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };
        const value = {
            0: 42,  // initial value from lastConnection
            1: 1,
        };
        const setFieldValue = (_name: string, _value: any) => { };
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();
        expect(screen.getByText(/Number of workers for CUDA device 1:/)).toBeInTheDocument();
        expect(screen.getAllByRole('spinbutton').length).toBe(2);
    });

    it('sets to zero on blur if empty', async () => {
        const config = deepcopy(defaultConfig);
        const user = userEvent.setup();

        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0],
            has_cupy: true,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };


        // let's say our field is empty...
        const value = {
            0: " "
        };
        const setFieldValue = vi.fn((_name, _value) => {})
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();

        const inp = screen.getByRole('spinbutton');
        // and we "blur" by tabbing away:
        await user.click(inp);
        await user.keyboard('{Tab}');

        expect(setFieldValue).toHaveBeenCalled();
        expect(setFieldValue).toHaveBeenCalledWith("gpus", {0: 0});
    });

    it('keeps the empty string as value', () => {
        const config = deepcopy(defaultConfig);

        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0],
            has_cupy: true,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };

        // we start with one worker:
        const value = {
            0: 1,
        };
        const setFieldValue = vi.fn((_name, _value) => {})
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();

        const inp = screen.getByRole('spinbutton');
        // and we fire a change event explicitly:
        fireEvent.change(inp, {target: {value: ''}});

        expect(setFieldValue).toHaveBeenCalled();
        expect(setFieldValue).toHaveBeenCalledWith("gpus", {0: ''});
    });

    it('properly reacts to change events with numbers', () => {
        const config = deepcopy(defaultConfig);

        config.devices = {
            cpus: [0, 1, 2, 3, 4, 5, 6],
            cudas: [0],
            has_cupy: true,
        };
        config.lastConnection = {
            type: ClusterTypes.LOCAL,
            address: 'tcp://localhost:8786',
            cudas: {0: 42},
        };

        // we start with one worker:
        const value = {
            0: 1,
        };
        const setFieldValue = vi.fn((_name, _value) => {})
        render(
            <GPUSelector config={config} name="gpus" value={value} setFieldValue={setFieldValue} />
        );

        expect(screen.getByText(/Number of workers for CUDA device 0:/)).toBeInTheDocument();

        const inp = screen.getByRole('spinbutton');
        // and we fire a change event explicitly:
        fireEvent.change(inp, {target: {value: 21}});

        expect(setFieldValue).toHaveBeenCalled();
        expect(setFieldValue).toHaveBeenCalledWith("gpus", {0: 21});
    });
})