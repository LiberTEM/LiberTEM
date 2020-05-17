import * as React from "react";
import { Button, Form } from "semantic-ui-react";
import { parseNumListWithPadding } from "../helpers";
import ScanSizePart from "./ScanSizePart";

interface ScanSizeProps {
    value: string,
    minScan: number,
    maxScan: number,
    setFieldValue: (field: string, value: any, shouldValidate?: boolean) => void,
}

const ScanSize: React.FC<ScanSizeProps> = ({ value, minScan, maxScan, setFieldValue }) => {

    const [scanSize, setScanSize] = React.useState(parseNumListWithPadding(value, minScan));

    const scanRefsArray = React.useRef<HTMLInputElement[]>([]);

    const scanSizeChangeHandle = (idx: number, value: string) => {
      const newScanSize = [...scanSize];
      newScanSize[idx] = value;
      setScanSize(newScanSize);
      setFieldValue("scan_size", newScanSize.toString());
    };

    const commaPressHandle = (idx: number, keyCode: number) => {
        if(keyCode === 188){
            if(idx===(scanSize.length-1)) {
              newScanDim();
            } else {
                scanRefsArray.current[idx+1].focus();
            }
        }
    }

    const newScanDim = () => {
      if(scanSize.length < maxScan) {
        const newScanSize = [...scanSize];
        newScanSize.push("");
        setScanSize(newScanSize);
        setFieldValue("scan_size", newScanSize.toString());
      }
    }

    React.useEffect(() => {
      if(scanSize.length > minScan) {
        scanRefsArray.current[scanSize.length-1].focus();
      }
    }, [scanSize.length, minScan]);

    const delScanDim = () => {
      if(scanSize.length > minScan) {
        const newScanSize = [...scanSize];
        newScanSize.pop();
        setScanSize(newScanSize);
        setFieldValue("scan_size", newScanSize.toString());
      }
    }

    return (
      <>
        <Form.Group>
          {scanSize.map((val, idx) => {
            const scanRef = (ref:HTMLInputElement) => { scanRefsArray.current[idx] = ref; }
            return <Form.Field width={2} key={idx}><ScanSizePart scanKey={idx} name={"scan_size_" + idx} id={"id_scan_size_" + idx} value={val} scanRef={scanRef} scanSizeChangeHandle={scanSizeChangeHandle} commaPressHandle={commaPressHandle} /></Form.Field>
          })}
          <Form.Field hidden={minScan === maxScan}>
            <Button onClick={newScanDim} disabled={scanSize.length === maxScan} type="button" icon="add" title="Add dimension" basic={true} color="blue" />
            <Button onClick={delScanDim} disabled={scanSize.length === minScan} type="button" icon="minus" title="Remove dimension" basic={true} color="blue" />
          </Form.Field>
        </Form.Group>
      </>
    );
}

export default ScanSize;
