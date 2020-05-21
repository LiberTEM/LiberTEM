import { FormikErrors, FormikValues, withFormik } from 'formik';
import * as pathfind from 'path';
import { DatasetTypes } from "../messages";
import { OpenFormProps } from "./types";
import { validateOpen } from "./validate";

export function parseNumList(nums: string) {
    return nums.split(",").filter(part => part.trim() !== "").map(part => +part);
}

export function parseNumListWithPadding(nums: string, minLength: number, maxLength: number) {
    let initialList = new Array(minLength).fill("");
    if(nums) {
      let value = nums.split(",");
      if(minLength > value.length) {
        initialList = [...value, ...Array(minLength - value.length).fill("")];
      } else {
          value = value.slice(0, maxLength);
          initialList = [...value];
      }
    }
    return initialList;
}

export function parseNumListProduct(nums: string) {
    return parseNumList(nums).reduce((a,b) => a * b, 1);
}

export function frameCalcForOffset(syncOffset: number, navShapeProduct: number, imageCount: number) {
    return {
        framesSkippedStart: Math.max(0, syncOffset),
        framesIgnoredEnd: Math.max(0, imageCount - navShapeProduct - syncOffset),
        framesInsertedStart: Math.abs(Math.min(0, syncOffset)),
        framesInsertedEnd: Math.max(0, navShapeProduct - imageCount + syncOffset),
    };
}

export function isSigShapeValid(sigShape: string, nativeSigShape: string) {
    return parseNumListProduct(sigShape) === parseNumListProduct(nativeSigShape);
}

export function isSyncOffsetValid(syncOffset: number, imageCount: number) {
    return -imageCount < syncOffset && syncOffset < imageCount;
}

export function validateSyncOffsetAndSigShape(nativeSigShape: number[] | undefined, sigShape: string, syncOffset: number, imageCount: number | undefined): FormikErrors<FormikValues> {
    const res: FormikErrors<FormikValues> = {};
    if (nativeSigShape && !isSigShapeValid(sigShape, nativeSigShape.toString())) {
        res.sig_shape = `must be of size: ${parseNumListProduct(nativeSigShape.toString())}`;
    }
    if(imageCount && !isSyncOffsetValid(syncOffset, imageCount)) {
        res.sync_offset = `must be in (-${imageCount}, ${imageCount})`;
    }
    return res;
}

export function getInitial<T extends object, K extends keyof T, V>(key: K, otherwise: V, values?: T): V | T[K] {
    if (!values) {
        return otherwise;
    }
    const res = values[key] !== undefined ? values[key] : otherwise;
    return res;
}

export function getInitialName<T extends object, K extends keyof T>(key: K, otherwise: string, values?: T): string | T[K] {
    if (!values) {
        return pathfind.basename(otherwise);
    }
    const res = values[key] !== undefined ? values[key] : pathfind.basename(otherwise);
    return res;
}

export function isKnownDatasetType(detectedType: string) {
    return (Object.keys(DatasetTypes).some((v) => v === detectedType) ? true : false);
}

type FormToJsonFn<SubmitParams, FormParams> = (inParams: FormParams, path: string) => SubmitParams;
type PropsToValuesFn<SubmitParams, FormParams, FormInfo> = (props: OpenFormProps<SubmitParams, FormInfo>) => FormParams;
type CustomValidationFn<SubmitParams, FormParams, FormInfo> = (inParams: FormParams, props: OpenFormProps<SubmitParams, FormInfo>) => FormikErrors<FormikValues>;

interface WithValidationOpts<SubmitParams extends object, FormParams, FormInfo> {
    formToJson: FormToJsonFn<SubmitParams, FormParams>,
    mapPropsToValues: PropsToValuesFn<SubmitParams, FormParams, FormInfo>,
    type: DatasetTypes,
    customValidation?: CustomValidationFn<SubmitParams, FormParams, FormInfo>
    // WrappedComponent: React.FunctionComponent<FormikProps<FormParams> & OpenFormProps<SubmitParams>>
}

export function withValidation<SubmitParams extends object, FormParams, FormInfo>(
    opts: WithValidationOpts<SubmitParams, FormParams, FormInfo>
) {
    return withFormik<OpenFormProps<SubmitParams, FormInfo>, FormParams, FormInfo>({
        mapPropsToValues: opts.mapPropsToValues,
        handleSubmit: (values, formikBag) => {
            const { onSubmit, path } = formikBag.props;
            const submitData = opts.formToJson(values, path);
            onSubmit(submitData);
            formikBag.setSubmitting(false);
        },
        validate: (values, props) => {
            return validateOpen(opts.type, opts.formToJson(values, props.path), opts.customValidation?.(values, props));
        },
        enableReinitialize: true,
        validateOnChange: true,
        validateOnBlur: true,
    });
}
