import * as React from "react";
import { useDispatch, useSelector } from 'react-redux';
import { DropdownProps, Header, Segment } from "semantic-ui-react";
import { v4 as uuid } from 'uuid';
import { assertNotReached } from "../../helpers";
import { DatasetFormParams, DatasetTypes } from '../../messages';
import { RootReducer } from "../../store";
import * as datasetActions from "../actions";
import { OpenDatasetState } from "../types";
import BLOParamsForm from "./BLOParamsForm";
import DatasetTypeSelect from "./DatasetTypeSelect";
import EMPADParamsForm from "./EMPADParamsForm";
import FRMS6ParamsForm from "./FRMS6ParamsForm";
import HDF5ParamsForm from "./HDF5ParamsForm";
import K2ISParamsForm from "./K2ISParamsForm";
import MIBParamsForm from "./MIBParamsForm";
import MRCParamsForm from "./MRCParamsForm";
import NPYParamsForm from "./NPYParamsForm";
import RawFileParamsForm from "./RawFileParamsForm";
import SEQParamsForm from "./SEQParamsForm";
import SERParamsForm from "./SERParamsForm";
import TVIPSParamsForm from "./TVIPSParamsForm";
import RawCSRParamsForm from "./RawCSRParamsForm";
import DMParamsForm from "./DMParamsForm";

/**
 * Get the initial selection for the dataset type dropdown. If we have a previous
 * user selection, we use it, but only if the reset button was not pressed. If it was,
 * we use the detected dataset type, falling back to RAW.
 *
 * @param didReset flag: was the reset button pressed?
 * @param openState complete OpenDatasetState instance
 */
const getDefaultDSType = (didReset: boolean, openState: OpenDatasetState) => {
    const { formCachedParams, formDetectedParams } = openState;
    if (didReset) {
        return formDetectedParams ? formDetectedParams.type : DatasetTypes.RAW;
    }
    if (formCachedParams) {
        return formCachedParams.type;
    }
    if (formDetectedParams) {
        return formDetectedParams.type;
    }
    return DatasetTypes.RAW;
}


/**
 * Get the initial form field values. If we have previously entered values, we
 * use these, otherwise we use the detected parameters. If the reset button was
 * clicked, we use the detected params, but keep the original name field,
 *
 * @param didReset flag: was the reset button pressed?
 * @param openState complete OpenDatasetState instance
 */

const getFormInitial = (didReset: boolean, openState: OpenDatasetState) => {
    const { formCachedParams, formDetectedParams } = openState;
    if (didReset) {
        if (formDetectedParams) {
            return {
                ...formDetectedParams,
                name: formCachedParams ? formCachedParams.name : "",
            };
        }
        return undefined;
    }
    if (formCachedParams) {
        // To handle deprecation of scan_size and detector_size, fix this after complete removal
        let newFormCachedParams = formCachedParams;
        if (formCachedParams.scan_size) {
            newFormCachedParams = {
                ...newFormCachedParams,
                nav_shape: formCachedParams.scan_size,
                scan_size: [],
            };
        }
        if (formCachedParams.detector_size) {
            newFormCachedParams = {
                ...newFormCachedParams,
                sig_shape: formCachedParams.detector_size,
                detector_size: [],
            };
        }
        if (!formCachedParams.nav_shape) {
            newFormCachedParams = {
                ...newFormCachedParams,
                nav_shape: formDetectedParams ? formDetectedParams.nav_shape : [],
            };
        }
        if (!formCachedParams.sig_shape) {
            newFormCachedParams = {
                ...newFormCachedParams,
                sig_shape: formDetectedParams ? formDetectedParams.sig_shape : [],
            };
        }
        return newFormCachedParams;
    } else {
        return formDetectedParams;
    }
}

const getFormInfo = (openState: OpenDatasetState) => {
    const { formDetectedInfo } = openState;
    if (formDetectedInfo) {
        return formDetectedInfo;
    }
    return undefined;
}

/**
 * Dispatch to specific dataset opening forms, including a selection of dataset type
 * via a dropdown.
 */
const DatasetOpen = () => {
    const dispatch = useDispatch();
    const openState = useSelector((state: RootReducer) => state.openDataset);

    const [didReset, setReset] = React.useState(false);
    const formInitial = getFormInitial(didReset, openState);
    const formInfo = getFormInfo(openState);
    const defaultType = getDefaultDSType(didReset, openState);
    const [datasetType, setDatasetType] = React.useState(defaultType);

    const doSetType = (e: React.SyntheticEvent, data: DropdownProps) => setDatasetType(data.value as DatasetTypes);
    // FIXME: find out how to make ts correctly correlate the types of FormComponent
    // and initial and replace the huge switch below with something like:
    // const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
    // const formComponentMap = {
    //     [DatasetTypes.HDF5]: HDF5ParamsForm,
    //     [DatasetTypes.RAW]: RawFileParamsForm,
    //     [DatasetTypes.MIB]: MIBParamsForm,
    //     [DatasetTypes.BLO]: BLOParamsForm,
    //     [DatasetTypes.K2IS]: K2ISParamsForm,
    //     [DatasetTypes.SER]: SERParamsForm,
    //     [DatasetTypes.FRMS6]: FRMS6ParamsForm,
    //     [DatasetTypes.EMPAD]: EMPADParamsForm,
    // }
    // const FormComponent = formComponentMap[datasetType];

    const renderForm = (form: React.ReactNode) => (
        <Segment>
            Type: <DatasetTypeSelect onClick={doSetType} currentType={datasetType} />
            <Header as="h2">Open: {openState.formPath}</Header>
            {form}
        </Segment>
    );

    const datasetTypeInfo = useSelector((state: RootReducer) => state.config.datasetTypes[datasetType])

    const commonParams = {
        path: openState.formPath,
        onSubmit: (params: DatasetFormParams) => {
            dispatch(datasetActions.Actions.create({
                id: uuid(),
                params,
            }))
        },
        onCancel: () => dispatch(datasetActions.Actions.cancelOpen()),
        onReset: () => {
            setReset(true);
            setDatasetType(getDefaultDSType(true, openState));
        },
        datasetTypeInfo,
    }

    switch (datasetType) {
        case DatasetTypes.HDF5: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<HDF5ParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.RAW: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<RawFileParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.MIB: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<MIBParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.NPY: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<NPYParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.BLO: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<BLOParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.K2IS: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<K2ISParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.SER: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<SERParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.FRMS6: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<FRMS6ParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.EMPAD: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<EMPADParamsForm {...commonParams} initial={initial} info={info} />)
        }
        case DatasetTypes.SEQ: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<SEQParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.MRC: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<MRCParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.TVIPS: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<TVIPSParamsForm {...commonParams} initial={initial} info={info} />);
        }
        case DatasetTypes.RAW_CSR: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<RawCSRParamsForm {...commonParams} initial={initial} info={info} />);
        }        
        case DatasetTypes.DM: {
            const initial = formInitial && datasetType === formInitial.type ? formInitial : undefined;
            const info = formInfo && datasetType === formInfo.type ? formInfo : undefined;
            return renderForm(<DMParamsForm {...commonParams} initial={initial} info={info} />);
        }
    }
    return assertNotReached("unknown dataset type");
}

export default DatasetOpen;
