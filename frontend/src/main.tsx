import React, { Suspense, lazy } from "react";
import ReactDOM from "react-dom/client";
import { Navigate, RouterProvider, createBrowserRouter } from "react-router-dom";
import { Layout } from "./components/Layout";
import "./styles.css";
import "./workbench.css";
import "./lab.css";

const SimulatePage = lazy(() => import("./pages/WorkbenchPage").then((module) => ({ default: module.WorkbenchPage })));
const ComparePage = lazy(() => import("./pages/ComparePage").then((module) => ({ default: module.ComparePage })));

const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage").then(m => ({ default: m.ExperimentsPage })));
const CasesPage = lazy(() => import("./pages/CasesPage").then(m => ({ default: m.CasesPage })));

function Deferred({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<div className="page-loading">正在装载工作台…</div>}>{children}</Suspense>;
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    errorElement: <div className="wb-loading"><h2>页面未能正常显示</h2><p>已保存的实验仍在服务器。请重新打开工作台；若问题持续，请使用配置导出文件检查输入。</p><a href="/simulate">重新打开工作台</a></div>,
    children: [
      { index: true, element: <Navigate to="/simulate" replace /> },
      { path: "/simulate", element: <Deferred><SimulatePage /></Deferred> },
      { path: "/experiments", element: <Deferred><ExperimentsPage /></Deferred> },
      { path: "/cases", element: <Deferred><CasesPage /></Deferred> },
      { path: "/compare", element: <Deferred><ComparePage /></Deferred> },
      { path: "/validate/ramp", element: <Navigate to="/compare" replace /> },
      { path: "/about", element: <Navigate to="/simulate" replace /> },
    ],
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><RouterProvider router={router} /></React.StrictMode>,
);
